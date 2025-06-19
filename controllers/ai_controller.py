##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## ai_controller
##

import numpy as np
from controllers.icontroller import IController
from car import Car
from camera_stream_server import CameraStreamServer
from logger import setup_logger, TimeLogger

logger = setup_logger(__name__)

class AIController(IController):
    """
    Controller for the AI model in the Robocar project.
    This controller handles the interaction with the AI model.
    """
    def __init__(self, car: Car, streaming: bool = False):
        """
        Initialize the AIController with a model.
        """
        from mask_generator.models.utils import load_pad_divisor_from_run_dir
        from mask_generator.trt_wrapper import TRTWrapper
        from mask_generator.transforms import KorniaInferTransform
        from racing.model import MyModel

        with TimeLogger("Import joblib torch and depthai", logger):
            import joblib
            import torch
            import depthai as dai

        self.torch = torch # Store torch reference
        self.dai = dai # Store depthai reference

        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.car = car
        self.streaming = streaming

        with TimeLogger("Initializing camera pipeline", logger):
            self.pipeline = self.init_camera()

        # Setup Racing Simulator
        model_path = "model24220ce995.joblib"

        with TimeLogger("Loading Racing Simulator model", logger):
            self.racing_model = MyModel(input_size=57, hidden_layers=[32, 64, 128, 64, 32], output_size=2).to(self.device)

        with TimeLogger(f"Loading racing model weights from {model_path}", logger):
            save_dict = joblib.load(model_path)
            self.racing_model.load_state_dict(save_dict["model_weights"])

        self.racing_model.eval()
        self.racing_scaler = save_dict["scaler"]

        # Setup Mask Generator
        with TimeLogger("Loading Mask Generator model", logger):
            pad_divisor = load_pad_divisor_from_run_dir("mask_generator/run")
            ENGINE_PATH = "mask_generator/run/model_fp16.engine"
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

        self.previous_data = {}

        self.camera_stream = None

    def init_camera(self):
        pipeline = self.dai.Pipeline()
        cam_color = pipeline.createColorCamera()
        cam_color.setPreviewSize(448, 256)
        cam_color.setInterleaved(False)
        cam_color.setColorOrder(self.dai.ColorCameraProperties.ColorOrder.BGR)

        xout = pipeline.createXLinkOut()
        xout.setStreamName("video")
        cam_color.preview.link(xout.input)

        return pipeline

    def get_rays_data(self, image: np.ndarray, generate_image: bool = False) -> tuple:
        """
        Generate rays data from the input image using the mask generator model.
        Args:
            image: Input image as a numpy array.
        Returns:
            Dictionary containing distances and rays.
        """
        from mask_generator.utils import get_mask
        from mask_generator.ray_generator import generate_rays_torch, generate_rays_vectorized, show_rays

        # with TimeLogger("Generating mask from image", logger):
        #     mask = get_mask(self.mask_model, self.mask_transform, image)
        with TimeLogger("Generating mask from image", logger):
            mask_tensor = get_mask(self.mask_model, self.mask_transform, image)

        # with TimeLogger("Generating rays from mask", logger):
        #     distances, ray_endpoints = generate_rays_vectorized(mask_tensor, num_rays=50, fov_degrees=120, max_distance=400)

        with TimeLogger("Generating rays from mask", logger):
            distances, ray_endpoints = generate_rays_torch(mask_tensor, num_rays=50, fov_degrees=120, max_distance=400, device=self.device)

        rays_image = None
        if generate_image:
            mask = mask_tensor.cpu().numpy()
            rays_image = show_rays(mask, ray_endpoints, distances, image, generate_image=True)

        return distances, rays_image if rays_image else None

    def get_data(self, image: np.ndarray, generate_image: bool = False) -> tuple:
        """
        Prepare the input data for the AI model.
        Args:
            image: Input image as a numpy array.
        Returns:
            Scaled input data as a dictionary.
        """
        rays_data, image_rays = self.get_rays_data(image, generate_image=generate_image)

        speed = self.car.get_speed() / 8 * 40 # Scale speed to a range of 0-40 # TEMPORARY, TRYING TO MATCH RACING SIMULATOR SPEED

        init_colomns = ["speed", "steering"] + [f"pos_{coord}" for coord in ['x', 'y', 'z']] \
                + [f"ray_{i}" for i in range(1, self.num_rays + 1)]

        data = {column: 0.0 for column in init_colomns}

        for i in range(self.num_rays):
            data[f"ray_{i+1}"] = float(rays_data[f"ray_{i}"])

        data["speed"] = speed

        data["delta_speed"] = data["speed"] - self.previous_data.get("speed", 0.0)
        data["delta_steering"] = data["steering"] - self.previous_data.get("steering", 0.0)

        ray_values = np.array([rays_data[f"ray_{i}"] / 400 * 250 for i in range(50)]) # TEMPORARY, TRYING TO MATCH RACING SIMULATOR RAY VALUES

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

        # Update previous data
        self.previous_data = data.copy()

        return data, image_rays

    def get_actions(self, data: dict) -> dict:
        input_data = [data[column] for column in self.input_columns]
        data_scaled = self.racing_scaler.transform([input_data])
        data_tensor = self.torch.tensor(data_scaled, dtype=self.torch.float32, device=self.device)

        with TimeLogger("Running racing model inference", logger):
            with self.torch.no_grad():
                prediction = self.racing_model(data_tensor).cpu().numpy().squeeze()

        logger.debug(f"Prediction: {prediction}")
        return {
            "throttle": prediction[0],
            "steering": prediction[1]
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

        with self.dai.Device(self.pipeline) as cam_device:
            video_queue = cam_device.getOutputQueue(name="video", maxSize=4, blocking=False)
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
                            in_video = video_queue.get()
                            frame = in_video.getCvFrame()
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
