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

    # Controller (default is gamepad)
    parser.add_argument(
        "--controller",
        choices=["gamepad", "ai"],
        default="gamepad",
        help="Select the controller type: 'gamepad' or 'ai'. Default is 'gamepad'."
    )

    # Camera stream option
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable camera stream for AI controller."
    )

    return parser.parse_args()


def main():
    print(f"Starting main")
    args = parse_args()

    car = Car(port=PORT, power_limit=0.03)
    controller_cls = {
        "gamepad": GamepadController,
        "ai": lambda car: AIController(car, streaming=args.stream)
    }[args.controller]

    controller = controller_cls(car)

    print(f"Using {args.controller} controller.")
    controller.run()

if __name__ == "__main__":
    main()
