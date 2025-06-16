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

class AIController(IController):
    """
    Controller for the AI model in the Robocar project.
    This controller handles the interaction with the AI model.
    """
    def __init__(self, car: Car, is_camera_stream: bool = False):
        """
        Initialize the AIController with a model.
        """
        import time

        from mask_generator.model_loader import load_model_from_run_dir
        from mask_generator.transforms import EvalTransform, TensorDecoder
        from racing.model import MyModel

        time_before_import = time.time()
        import joblib
        import torch
        print(f"Time taken to import joblib and torch: {time.time() - time_before_import:.2f} seconds")
        import depthai as dai

        self.torch = torch # Store torch reference
        self.dai = dai # Store depthai reference
        print(f"Time taken to import modules: {time.time() - time_before_import:.2f} seconds")

        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.car = car
        self.is_camera_stream = is_camera_stream

        time_before_camera = time.time()
        self.pipeline = self.init_camera()
        print(f"Time taken to initialize camera: {time.time() - time_before_camera:.2f} seconds")
        # Setup Racing Simulator
        model_path = "model24220ce995.joblib"

        time_before_model = time.time()
        self.racing_model = MyModel(input_size=57, hidden_layers=[32, 64, 128, 64, 32], output_size=2).to(self.device)
        print(f"Time taken to initialize model: {time.time() - time_before_model:.2f} seconds")

        time_before_load = time.time()
        save_dict = joblib.load(model_path)
        print(f"Time taken to load model weights: {time.time() - time_before_load:.2f} seconds")

        time_before_load_weights = time.time()
        self.racing_model.load_state_dict(save_dict["model_weights"])
        self.racing_model.eval()
        self.racing_scaler = save_dict["scaler"]
        print(f"Time taken to load model: {time.time() - time_before_load_weights:.2f} seconds")

        # Setup Mask Generator
        time_before_mask = time.time()
        self.mask_model, pad_divisor = load_model_from_run_dir("mask_generator/best_run", self.device)
        print(f"Time taken to load mask generator model: {time.time() - time_before_mask:.2f} seconds")

        time_before_mask_transform = time.time()
        self.mask_transform = EvalTransform(pad_divisor=pad_divisor, to_tensor=True)
        self.mask_decoder = TensorDecoder()
        print(f"Time taken to initialize mask generator transform: {time.time() - time_before_mask_transform:.2f} seconds")

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
        cam_color.setPreviewSize(455, 256)
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
        from mask_generator.model_inference import infer_mask
        from mask_generator.ray_generator import generate_rays, show_rays

        mask = infer_mask(self.mask_model, transform=self.mask_transform, decoder=self.mask_decoder, image=image, device=self.device)
        distances, ray_endpoints = generate_rays(mask, num_rays=50, fov_degrees=120, max_distance=400)

        if generate_image:
            rays_image = show_rays(mask, ray_endpoints, distances, image, num_rays=self.num_rays, fov_degrees=self.fov, max_distance=400, generate_image=True)

        return distances, rays_image if generate_image else None

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
        """  """
        input_data = [data[column] for column in self.input_columns]
        print(f"Input data: {input_data}")
        data_scaled = self.racing_scaler.transform([input_data])
        print(f"Scaled data: {data_scaled}")
        data_tensor = self.torch.tensor(data_scaled, dtype=self.torch.float32, device=self.device)

        with self.torch.no_grad():
            prediction = self.racing_model(data_tensor).cpu().numpy().squeeze()

        print(f"Prediction: {prediction}")
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

            with CameraStreamServer(on=self.is_camera_stream) as stream:
                self.camera_stream = stream

                while True:
                    now = time.time()
                    delta = now - prev_time
                    prev_time = now
                    if delta > 0:
                        fps_history.append(1.0 / delta)
                    avg_fps = sum(fps_history) / len(fps_history)
                    print(f"\rAverage FPS: {avg_fps:.2f}  ", end='')
                    in_video = video_queue.get()
                    frame = in_video.getCvFrame()
                    image_rgb = cvtColor(frame, COLOR_BGR2RGB)

                    data, image_rays = self.get_data(image_rgb, generate_image=self.is_camera_stream)

                    # STREAMING
                    if self.camera_stream is not None:
                        self.camera_stream.stream_image(image_rays)

                    actions = self.get_actions(data)
                    self.car.set_actions(actions)
