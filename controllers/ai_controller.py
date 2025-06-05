##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## ai_controller
##

import torch
import pickle
import joblib
import depthai as dai
from collections import deque
import cv2
import time
import numpy as np
from controllers.icontroller import IController
from racing.model import MyModel
from car import Car
from mask_generator.model_loader import load_model_from_run_dir
from mask_generator.model_inference import infer_mask
from mask_generator.transforms import EvalTransform, TensorDecoder
from mask_generator.ray_generator import generate_rays

class AIController(IController):
    """
    Controller for the AI model in the Robocar project.
    This controller handles the interaction with the AI model.
    """

    def __init__(self, car: Car):
        """
        Initialize the AIController with a model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.car = car
        self.pipeline = self.init_camera()

        # Setup Racing Simulator
        model_path = "model24220ce995.joblib"
        self.racing_model = MyModel(input_size=57, hidden_layers=[32, 64, 128, 64, 32], output_size=2).to(self.device)
        save_dict = joblib.load(model_path)
        self.racing_model.load_state_dict(save_dict["model_weights"])
        self.racing_model.eval()
        self.racing_scaler = save_dict["scaler"]

        # Setup Mask Generator
        self.mask_model, pad_divisor = load_model_from_run_dir("mask_generator/best_run", self.device)
        self.mask_transform = EvalTransform(pad_divisor=pad_divisor, to_tensor=True)
        self.mask_decoder = TensorDecoder()

    def init_camera(self):
        pipeline = dai.Pipeline()
        cam_color = pipeline.createColorCamera()
        cam_color.setPreviewSize(455, 256)
        cam_color.setInterleaved(False)
        cam_color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

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
        mask = infer_mask(self.mask_model, transform=self.mask_transform, decoder=self.mask_decoder, image=image, device=self.device)
        distances, _ = generate_rays(mask, num_rays=50, fov_degrees=120)
        return distances

    def get_input_data(self, image: np.ndarray) -> dict:
        """
        Prepare the input data for the AI model.
        Args:
            image: Input image as a numpy array.
        Returns:
            Scaled input data as a dictionary.
        """
        rays_data = self.get_rays_data(image)
        return {}

    def get_actions(self, input_data: np.ndarray) -> dict:

        data_scaled = self.racing_scaler.transform([input_data])
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction = self.model(data_tensor).numpy().squeeze()

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
        with dai.Device(self.pipeline) as cam_device:
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
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                input_data = self.get_input_data(image_rgb)
                actions = self.get_actions(input_data)
                self.car.set_actions(actions)
