import signal
import sys

from gamepad_inputs import GamepadInput
from car import Car

PORT = "/dev/ttyACM0"

should_exit = False

def handle_sigterm(signum, frame):
    global should_exit
    print(f"Received signal {signum}, preparing to exit...")
    should_exit = True

def handle_events(updated, car):
    """
    Handle the updated gamepad events.

    :param updated: List of tuples containing the event code and state.
    """
    for code, state in updated:
        if code == 'ABS_Y':
            print(f"Left Joystick Vertical Axis: {state}")
        elif code == 'ABS_X':
            print(f"Left Joystick Horizontal Axis: {state}")
            position = (state + 32768) / 65535.0
            position = max(0.0, min(1.0, position))  # Clamp to [0.0, 1.0]
            car.set_servo(position)
        elif code == 'BTN_SOUTH':
            print("Button South pressed")
        elif code == 'BTN_WEST':
            print("Button North pressed")
        elif code == 'BTN_EAST':
            print("Button East pressed")
        elif code == 'BTN_NORTH':
            print("Button West pressed")
        elif code == 'BTN_TL':
            print("Left Trigger pressed")
        elif code == 'BTN_TR':
            print("Right Trigger pressed")
        elif code == 'BTN_THUMBL':
            print("Left Thumb pressed")
        elif code == 'BTN_THUMBR':
            print("Right Thumb pressed")
        elif code == 'ABS_RX':
            print(f"Right Joystick Horizontal Axis: {state}")
        elif code == 'ABS_RY':
            print(f"Right Joystick Vertical Axis: {state}")
        elif code == 'ABS_Z':
            print(f"Left Trigger Axis: {state}")
            car.set_duty_cycle(-state / 255.0)
        elif code == 'ABS_RZ':
            print(f"Right Trigger Axis: {state}")
            car.set_duty_cycle(state / 255.0)
        elif code == 'ABS_HAT0Y':
            car.increment_power_limit(-state / 100)
            print(f"New power limit: {car.get_power_limit()}")
        else:
            print(f"Unhandled event: {code} with state {state}")

import random

def main():
    gamepad_input = GamepadInput()
    car = Car(port=PORT, power_limit=0.1)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    try:
        while not should_exit:
            updated = gamepad_input.update()
            handle_events(updated, car)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Exiting gracefully...")
        car.set_duty_cycle(0)
        print("Stopping car motors.")
        car.set_servo(0.5)
        print("Resetting servo to neutral position.")

if __name__ == "__main__":
    main()
    print("Gamepad input handling finished.")
    sys.exit(0)
    print("Exiting program.")
