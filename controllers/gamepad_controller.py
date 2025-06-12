##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## gamepad_controller
##

from controllers.icontroller import IController
from inputs import get_gamepad
from car import Car
from camera_stream_server import CameraStreamServer

class GamepadController(IController):
    """
    Controller for handling gamepad inputs in the Robocar project.
    This controller reads the state of the gamepad and provides methods to access it.
    """

    def __init__(self, car: Car):
        """
        Initialize the GamepadController.
        """
        self.car = car
        self.gamepad_state = {}
        self.updated = []
        self.old_state = {'throttle': 0.0, 'steering': 0.5}

    def update(self):
        """Update the gamepad state by reading the current inputs."""
        events = get_gamepad()
        updated = []
        for event in events:
            if event.ev_type in ('Key', 'Absolute'):
                prev_state = self.gamepad_state.get(event.code, 0)
                self.gamepad_state[event.code] = event.state
                if prev_state != event.state:
                    updated.append((event.code, event.state))
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

        for code, state in self.updated:
            if code == 'ABS_X':
                action['steering'] = self.get_steering((state + 32768) / 65535.0)
            elif code == 'ABS_Z':
                action['throttle'] = self.get_throttle(-state / 255.0)
            elif code == 'ABS_RZ':
                action['throttle'] = self.get_throttle(state / 255.0)

        self.old_state = action.copy()
        return action

    def run(self):
        """
        Run the gamepad controller with the given car instance.
        This method continuously updates the gamepad state and applies the actions to the car.
        """
        import cv2
        import numpy as np
        with CameraStreamServer() as camera_server:
            while True:
                updated = self.update()
                if updated:
                    actions = self.get_actions()
                    self.car.set_actions(actions)
                    fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2_image = cv2.cvtColor(fake_image, cv2.COLOR_BGR2RGB)
                    camera_server.stream_image(cv2_image)
