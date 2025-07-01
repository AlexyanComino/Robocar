##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## ai_controller
##

import numpy as np
from controllers.icontroller import IController
from car import Car
from camera import Camera
from camera_stream_server import CameraStreamServer
from logger import setup_logger, TimeLogger

logger = setup_logger(__name__)

class AIController(IController):
    """
    Controller for the AI model in the Robocar project.
    This controller handles the interaction with the AI model.
    """
    def __init__(self, car: Car, mask_model_dir: str, streaming: bool = False):
        """ Initialize the AIController with a model. """
        with TimeLogger("Import necessary modules", logger):
            from mask_generator.models.utils import load_pad_divisor_from_run_dir
            from mask_generator.trt_wrapper import TRTWrapper
            from mask_generator.transforms import KorniaInferTransform
            from racing.model import MyModel

            import torch

        self.torch = torch # Store torch reference

        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.car = car
        self.streaming = streaming

        # Setup Racing Simulator
        # Racing Simulator data
        self.fov = 160
        self.num_rays = 50

        racing_model_path = "racing/models/modele3e9af0268.pth"

        self.input_columns = ['speed', 'delta_speed', 'delta_steering', 'angle_closest_ray', 'avg_ray', 'std_ray', 'min_ray', 'max_ray',
                              'avg_ray_left', 'avg_ray_center', 'avg_ray_right',
                              'ray_balance', 'acceleration'] + [f"ray_{i}" for i in range(1, self.num_rays + 1)]

        self.output_columns = ["input_speed", "input_steering"]

        with TimeLogger("Loading Racing Simulator model", logger):
            self.racing_model = MyModel(input_size=len(self.input_columns), hidden_layers=[16, 32, 64, 32, 16], output_size=2).to(self.device)

        with TimeLogger(f"Loading racing model weights from {racing_model_path}", logger):
            self.racing_model.load_state_dict(torch.load(racing_model_path, map_location=self.device))

        with TimeLogger("Setting Racing model to evaluation mode", logger):
            self.racing_model.eval()
            example_input = torch.randn(len(self.input_columns), device=self.device)
            self.racing_model = torch.jit.trace(self.racing_model, example_input)

        # Setup Mask Generator
        with TimeLogger("Loading Mask Generator model", logger):
            pad_divisor = load_pad_divisor_from_run_dir(mask_model_dir)
            engine_path = f"{mask_model_dir}/model_fp16.engine"
            logger.info(f"Using Mask Generator engine: {engine_path}")
            self.mask_model = TRTWrapper(engine_path, device=self.device)
            self.height, self.width = self.mask_model.get_input_shape()[2:]
            logger.info(f"Mask Generator width: {self.width}, height: {self.height}")

        with TimeLogger("Initializing mask generator transform", logger):
            self.mask_transform = KorniaInferTransform(
                pad_divisor=pad_divisor,
                image_size=(self.height, self.width),
                device=self.device
            )

        self.previous_data = {}

        self.camera_stream = None

    def get_rays_data(self, image: np.ndarray, generate_image: bool = False) -> tuple:
        """
        Generate rays data from the input image using the mask generator model.
        Args:
            image: Input image as a numpy array.
        Returns:
            Dictionary containing distances and rays.
        """
        from mask_generator.utils import get_mask
        from mask_generator.ray_generator import generate_rays_vectorized, show_rays

        with TimeLogger("Generating mask from image", logger):
            mask = get_mask(self.mask_model, self.mask_transform, image)

        with TimeLogger("Generating rays from mask", logger):
            distances, ray_endpoints = generate_rays_vectorized(mask, num_rays=self.num_rays, fov_degrees=self.fov)

        rays_image = None
        if generate_image:
            with TimeLogger("Showing rays on image", logger):
                rays_image = show_rays(mask, ray_endpoints, distances, image, generate_image=True)

        return distances, rays_image

    def get_data(self, image: np.ndarray, generate_image: bool = False) -> tuple:
        """
        Prepare the input data for the AI model.
        Args:
            image: Input image as a numpy array.
        Returns:
            Scaled input data as a dictionary.
        """
        data, image_rays = self.get_rays_data(image, generate_image=generate_image)

        with TimeLogger("Calculating features", logger):
            # For Power limit 0.03 is 1.55
            # For Power limit 0.02 is 0.71
            speed = self.car.get_speed() / 1.55
            speed = max(0.0, min(speed, 1.0))  # Clamp speed to [0, 1]

            data["speed"] = speed

            data["delta_speed"] = data["speed"] - self.previous_data.get("speed", 0.0)
            data["delta_steering"] = data["steering"] - self.previous_data.get("steering", 0.0)

            # List of ray values
            ray_values = np.array([data[f"ray_{i + 1}"] for i in range(self.num_rays)])

            # Find the closest ray to the car
            closest_ray_index = np.argmin(ray_values)
            angle_step = self.fov / (self.num_rays - 1)
            data["angle_closest_ray"] = -(self.fov / 2) + closest_ray_index * angle_step

            # Calculate the average, standard deviation, min, and max of the ray values
            data["avg_ray"] = np.mean(ray_values)
            data["std_ray"] = np.std(ray_values)
            data["min_ray"] = np.min(ray_values)
            data["max_ray"] = np.max(ray_values)

            left_indices = range(self.num_rays // 3)
            center_indices = range(self.num_rays // 3, 2 * self.num_rays // 3)
            right_indices = range(2 * self.num_rays // 3, self.num_rays)

            data["avg_ray_left"] = np.mean(ray_values[list(left_indices)])
            data["avg_ray_center"] = np.mean(ray_values[list(center_indices)])
            data["avg_ray_right"] = np.mean(ray_values[list(right_indices)])

            data["ray_balance"] = data["avg_ray_right"] - data["avg_ray_left"]

            # Acceleration calculation
            prev_delta_speed = self.previous_data.get("delta_speed", 0.0)
            data["acceleration"] = data["delta_speed"] - prev_delta_speed

            # Update state
            self.previous_data = data.copy()

        return data, image_rays

    def get_actions(self, data: dict) -> dict:
        # Avoid list + tensor creation overhead
        input_array = np.fromiter((data[col] for col in self.input_columns), dtype=np.float32)
        data_tensor = self.torch.from_numpy(input_array).to(self.device)

        with TimeLogger("Running racing model inference", logger):
            with self.torch.inference_mode():
                prediction = self.racing_model(data_tensor).detach().cpu().numpy()

        logger.debug(f"Prediction: {prediction}")
        return {
            "throttle": float(prediction[0]),
            "steering": float(prediction[1])
        }

    def run(self):
        """
        Run the AI controller with the given car instance.
        This method continuously updates the AI state and applies the actions to the car.
        Args:
            car: An instance of the Car class to control.
        """

        import time
        from collections import deque
        from cv2 import cvtColor, COLOR_BGR2RGB

        with Camera(width=self.width, height=self.height) as camera:
            fps_history = deque(maxlen=30)

            prev_time = time.time()

            with CameraStreamServer(on=self.streaming) as stream:
                self.camera_stream = stream

                while True:
                    now = time.time()
                    delta = now - prev_time
                    prev_time = now
                    if delta > 0:
                        fps_history.append(1.0 / delta)
                    avg_fps = sum(fps_history) / len(fps_history)
                    logger.debug(f"Average FPS: {avg_fps:.2f}")
                    with TimeLogger("Processing video frame", logger):
                        with TimeLogger("Getting video frame from queue", logger):
                            frame = camera.get_frame()
                            image_rgb = cvtColor(frame, COLOR_BGR2RGB)

                        with TimeLogger("Getting data from image", logger):
                            data, image_rays = self.get_data(image_rgb, generate_image=self.streaming)

                        # STREAMING
                        if self.camera_stream is not None:
                            with TimeLogger("Streaming image", logger):
                                self.camera_stream.stream_image(image_rays)

                        with TimeLogger("Getting actions from data", logger):
                            actions = self.get_actions(data)
                        with TimeLogger("Setting actions to car", logger):
                            self.car.set_actions(actions)
