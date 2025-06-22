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
    def __init__(self, car: Car, streaming: bool = False):
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
        model_path = "model6eba09feab.pth"

        with TimeLogger("Loading Racing Simulator model", logger):
            self.racing_model = MyModel(input_size=57, hidden_layers=[32, 64, 128, 64, 32], output_size=2).to(self.device)

        with TimeLogger(f"Loading racing model weights from {model_path}", logger):
            self.racing_model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.racing_model.eval()

        example_input = torch.randn(57, device=self.device)
        self.racing_model = torch.jit.trace(self.racing_model, example_input)

        # Setup Mask Generator
        with TimeLogger("Loading Mask Generator model", logger):
            pad_divisor = load_pad_divisor_from_run_dir("mask_generator/run")
            ENGINE_PATH = "mask_generator/run/model_fp16_256x768.engine"
            self.mask_model = TRTWrapper(ENGINE_PATH, device=self.device)

        with TimeLogger("Initializing mask generator transform", logger):
            self.mask_transform = KorniaInferTransform(
                pad_divisor=pad_divisor,
                device=self.device
            )

        # Racing Simulator data
        self.fov = 120
        self.max_rays = 50
        self.num_rays = self.max_rays

        self.input_columns = ['speed', 'delta_speed', 'angle_closest_ray',
                              'avg_ray_left', 'avg_ray_center', 'avg_ray_right',
                              'ray_balance'] + [f"ray_{i}" for i in range(1, 51)]
        self.output_columns = ["input_speed", "input_steering"]

        self.init_columns = ["speed", "steering"] + [f"pos_{coord}" for coord in ['x', 'y', 'z']] \
                    + [f"ray_{i}" for i in range(1, self.num_rays + 1)]

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
        rays_data, image_rays = self.get_rays_data(image, generate_image=generate_image)

        with TimeLogger("Calculating features", logger):
            speed = self.car.get_speed() / 1.55

            data = dict.fromkeys(self.init_columns, 0.0)

            ray_values = np.fromiter((rays_data[f"ray_{i}"] for i in range(self.num_rays)), dtype=float)

            for i in range(self.num_rays):
                data[f"ray_{i+1}"] = ray_values[i]

            # Basic data
            data["speed"] = speed
            data["delta_speed"] = speed - self.previous_data.get("speed", 0.0)
            data["delta_steering"] = data["steering"] - self.previous_data.get("steering", 0.0)

            # Closest ray and angle
            closest_ray_index = ray_values.argmin()
            angle_step = self.fov / (self.num_rays - 1)
            data["angle_closest_ray"] = -self.fov / 2 + closest_ray_index * angle_step

            # Basic stats
            data["avg_ray"] = ray_values.mean()
            data["std_ray"] = ray_values.std()
            data["min_ray"] = ray_values.min()
            data["max_ray"] = ray_values.max()

            n = self.num_rays
            third = n // 3
            data["avg_ray_left"] = ray_values[:third].mean()
            data["avg_ray_center"] = ray_values[third:2*third].mean()
            data["avg_ray_right"] = ray_values[2*third:].mean()
            data["ray_balance"] = data["avg_ray_right"] - data["avg_ray_left"]

            # Acceleration
            data["acceleration"] = data["delta_speed"] - self.previous_data.get("delta_speed", 0.0)

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

        with Camera() as camera:
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
                    # print(f"\rAverage FPS: {avg_fps:.2f}  ", end='')
                    with TimeLogger("Processing video frame", logger):
                        with TimeLogger("Getting video frame from queue", logger):
                            frame = camera.get_frame()
                            image_rgb = cvtColor(frame, COLOR_BGR2RGB)

                        with TimeLogger("Getting data from image", logger):
                            data, image_rays = self.get_data(image_rgb, generate_image=self.streaming)

                        # STREAMING
                        if self.camera_stream is not None:
                            self.camera_stream.stream_image(image_rays)

                        with TimeLogger("Getting actions from data", logger):
                            actions = self.get_actions(data)
                        with TimeLogger("Setting actions to car", logger):
                            self.car.set_actions(actions)
