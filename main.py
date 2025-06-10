##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## main
##

from argparse import ArgumentParser

from controllers.gamepad_controller import GamepadController
from controllers.ai_controller import AIController
from car import Car

PORT = "/dev/ttyACM0"

def parse_args():
    parser = ArgumentParser(description="Robocar Project")
    parser.add_argument(
        "--controller",
        choices=["gamepad", "ai"],
        default="gamepad",
        help="Select the controller type: 'gamepad' or 'ai'. Default is 'gamepad'."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    car = Car(port=PORT, power_limit=0.05)
    if args.controller == "gamepad":
        controller = GamepadController(car)
    elif args.controller == "ai":
        controller = AIController(car)
    else:
        raise ValueError("Invalid controller type. Choose 'gamepad' or 'ai'.")
    print(f"Using {args.controller} controller.")

    controller.run()

if __name__ == "__main__":
    main()
