##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## ai_controller
##

import numpy as np

from controllers.icontroller import IController
from car import Car

class AIController(IController):
    """
    Controller for the AI model in the Robocar project.
    This controller handles the interaction with the AI model.
    """
    def __init__(self, car: Car):
        """
        Initialize the AIController with a model.
        """
        from mask_generator.model_loader import load_model_from_run_dir
        from mask_generator.transforms import EvalTransform, TensorDecoder
        from racing.model import MyModel
        import joblib

        import torch
        import depthai as dai
        import time

        time_before_import = time.time()
        self.torch = torch # Store torch reference
        self.dai = dai # Store depthai reference
        print(f"Time taken to import modules: {time.time() - time_before_import:.2f} seconds")

        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.car = car

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

    def get_rays_data(self, image: np.ndarray) -> dict:
        """
        Generate rays data from the input image using the mask generator model.
        Args:
            image: Input image as a numpy array.
        Returns:
            Dictionary containing distances and rays.
        """
        from mask_generator.model_inference import infer_mask
        from mask_generator.ray_generator import generate_rays

        mask = infer_mask(self.mask_model, transform=self.mask_transform, decoder=self.mask_decoder, image=image, device=self.device)
        distances, _ = generate_rays(mask, num_rays=50, fov_degrees=120, max_distance=300)
        return distances

    def get_input_data(self, image: np.ndarray) -> list:
        """
        Prepare the input data for the AI model.
        Args:
            image: Input image as a numpy array.
        Returns:
            Scaled input data as a dictionary.
        """
        rays_data = self.get_rays_data(image)
        old_speed = self.car.get_old_speed()
        speed = self.car.get_speed() / 8 * 40 # Scale speed to a range of 0-40
        delta_speed = speed - old_speed

        ray_values = np.array([rays_data[f"ray_{i}"] for i in range(50)])

        # Find the closest ray to the car
        closest_ray_index = np.argmin(ray_values)
        angle_step = 120 / (50 - 1)
        angle_closest_ray = -(120 / 2) + closest_ray_index * angle_step

        left_indices = range(50 // 3)
        center_indices = range(50 // 3, 2 * 50 // 3)
        right_indices = range(2 * 50 // 3, 50)

        avg_ray_left = np.mean(ray_values[list(left_indices)])
        avg_ray_center = np.mean(ray_values[list(center_indices)])
        avg_ray_right = np.mean(ray_values[list(right_indices)])

        ray_balance = avg_ray_right - avg_ray_left

        input_data = [speed, delta_speed, angle_closest_ray, avg_ray_left, avg_ray_center, avg_ray_right, ray_balance] + list(rays_data.values())
        return input_data

    def get_actions(self, input_data: list) -> dict:

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

        from cv2 import cvtColor, COLOR_BGR2RGB
        import time
        from collections import deque

        with self.dai.Device(self.pipeline) as cam_device:
            video_queue = cam_device.getOutputQueue(name="video", maxSize=4, blocking=False)
            fps_history = deque(maxlen=30)

            prev_time = time.time()

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

                input_data = self.get_input_data(image_rgb)
                actions = self.get_actions(input_data)
                self.car.set_actions(actions)
