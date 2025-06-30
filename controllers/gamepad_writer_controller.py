##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## gamepad_writer_controller
##

from controllers.icontroller import IController
from inputs import devices, UnpluggedError
from car import Car
from camera import Camera
from camera_stream_server import CameraStreamServer
from logger import setup_logger, TimeLogger
from data_recorder import DataRecorder

import numpy as np
from evdev import InputDevice, categorize, ecodes
import select

logger = setup_logger(__name__)

class GamepadWriterController(IController):
    """
    Controller for handling gamepad inputs in the Robocar project.
    This controller reads the state of the gamepad and provides methods to access it.
    """

    def __init__(self, car: Car, mask_model_dir: str, streaming: bool = False):
        """
        Initialize the GamepadController.
        """
        with TimeLogger("Import necessary modules", logger):
            from mask_generator.models.utils import load_pad_divisor_from_run_dir
            from mask_generator.trt_wrapper import TRTWrapper
            from mask_generator.transforms import KorniaInferTransform
            import torch

        self.torch = torch # Store torch reference

        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.car = car
        self.streaming = streaming

        self.gamepad_state = {}
        self.updated = []
        self.old_state = {'throttle': 0.0, 'steering': 0.5}

        self.gamepad = InputDevice('/dev/input/event2')

        # Setup Mask Generator
        with TimeLogger("Loading Mask Generator model", logger):
            pad_divisor = load_pad_divisor_from_run_dir(mask_model_dir)
            engine_path = f"{mask_model_dir}/model_fp16.engine"
            logger.info(f"Using Mask Generator engine: {engine_path}")
            self.mask_model = TRTWrapper(engine_path, device=self.device)
            self.height, self.width = self.mask_model.get_input_shape()[2:]
            logger.info(f"Mask Generator width: {self.width}, height: {self.height}")

        # with TimeLogger("Initializing mask generator transform", logger):
        #     self.mask_transform = KorniaInferTransform(
        #         pad_divisor=pad_divisor,
        #         image_size=(self.height, self.width),
        #         device=self.device
        #     )

        # Racing Simulator data
        self.fov = 160
        self.num_rays = 50


        self.input_columns = ['speed', 'delta_speed', 'angle_closest_ray',
                              'avg_ray_left', 'avg_ray_center', 'avg_ray_right',
                              'ray_balance'] + [f"ray_{i}" for i in range(1, self.num_rays + 1)]

        self.output_columns = ["input_speed", "input_steering"]

        self.init_columns = ["speed", "steering"] \
                    + [f"ray_{i}" for i in range(1, self.num_rays + 1)]

        self.previous_data = {}

        self.camera_stream = None

        self.data_recorder = DataRecorder(self.num_rays, self.fov)

    def get_rays_data(self, image: np.ndarray, generate_image: bool = False) -> tuple:
        """
        Generate rays data from the input image using the mask generator model.
        Args:
            image: Input image as a numpy array.
        Returns:
            Dictionary containing distances and rays.
        """
        from mask_generator.utils import get_mask
        from mask_generator.ray_generator import generate_rays, show_rays

        with TimeLogger("Generating mask from image", logger):
            mask = get_mask(self.mask_model, self.mask_transform, image)

        with TimeLogger("Generating rays from mask", logger):
            distances, ray_endpoints = generate_rays(mask, num_rays=self.num_rays, fov_degrees=self.fov)

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
            # For Power limit 0.03 is 1.55
            # For Power limit 0.02 is 0.71
            speed = self.car.get_speed() / 1.55
            speed = max(0.0, min(speed, 1.0))  # Clamp speed to [0, 1]

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

    def update(self):
        """
        Poll the evdev gamepad in a non-blocking way.
        """
        updated = []
        try:
            # Use select to check if data is ready, timeout 0 = non-blocking
            rlist, _, _ = select.select([self.gamepad], [], [], 0)
            if rlist:
                for event in self.gamepad.read():
                    # We only care about Key (buttons) and Absolute (axes) events
                    if event.type in (ecodes.EV_KEY, ecodes.EV_ABS):
                        # Normalize event code name (optional, can just use event.code)
                        code = ecodes.bytype[event.type].get(event.code, event.code)

                        prev_state = self.gamepad_state.get(event.code, 0)
                        if prev_state != event.value:
                            self.gamepad_state[event.code] = event.value
                            updated.append((code, event.value))

        except OSError as e:
            # Device might be disconnected or unavailable
            print(f"Gamepad read error or disconnected: {e}")

        self.updated = updated
        return updated


    def get_state(self, code: str) -> int:
        """Get the current state of a specific gamepad input."""
        return self.gamepad_state.get(code, 0)

    def get_steering(self, steering: float) -> float:
        """
        Get the steering value from the gamepad state.

        :return: Steering value in the range [0.0, 1.0].
        """
        return max(0.0, min(1.0, steering)) # Clamp to [0.0, 1.0]

    def get_throttle(self, throttle: float) -> float:
            """ Get the throttle value from the gamepad state. """
            return max(-1.0, min(1.0, throttle))  # Clamp to [-1.0, 1.0]

    def get_actions(self) -> dict:
        """
        Get the actions to be performed by the car based on the gamepad state.
        This method should be implemented to return a dictionary of actions.

        :return: A dictionary containing the actions derived from the gamepad state.
        """
        action = self.old_state.copy()

        print(self.updated)
        for code, state in self.updated:
            if code == 'ABS_X':
                action['steering'] = self.get_steering((state + 32768) / 65535.0)
                print(f"Steering: {action['steering']}")
            elif code == 'ABS_Z':
                action['throttle'] = self.get_throttle(-state / 255.0)
            elif code == 'ABS_RZ':
                action['throttle'] = self.get_throttle(state / 255.0)

        self.old_state = action.copy()
        return action

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
            fps_history = deque(maxlen=15)

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
                    # logger.debug(f"Average FPS: {avg_fps:.2f}")
                    # print(f"\rAverage FPS: {avg_fps:.2f}  ", end='')
                    # with TimeLogger("Processing video frame", logger):
                        # with TimeLogger("Getting video frame from queue", logger):
                        #     frame = camera.get_frame()
                        #     image_rgb = cvtColor(frame, COLOR_BGR2RGB)

                        # with TimeLogger("Getting data from image", logger):
                        #     data, image_rays = self.get_data(image_rgb, generate_image=self.streaming)

                        # # STREAMING
                        # if self.camera_stream is not None:
                        #     with TimeLogger("Streaming image", logger):
                        #         self.camera_stream.stream_image(image_rays)

                        # with TimeLogger("Getting actions from data", logger):
                    updated = self.update()
                    if updated:
                        actions = self.get_actions()
                        with TimeLogger("Setting actions to car", logger):
                            self.car.set_actions(actions)

                        # with TimeLogger("Writing data to recorder", logger):
                        #     self.data_recorder.write_data(data, self.old_state['throttle'], self.old_state['steering'])

