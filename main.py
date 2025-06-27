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

    subparsers = parser.add_subparsers(dest="controller", required=True, help="Controller type")

    gamepad_parser = subparsers.add_parser("gamepad", help="Control the car with a gamepad")

    ai_parser = subparsers.add_parser("ai", help="Control the car with AI model")
    ai_parser.add_argument(
        "--mask-model",
        type=str,
        required=True,
        help="Path to the mask generator model directory."
    )
    ai_parser.add_argument(
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
        "ai": lambda car: AIController(car, mask_model_dir=args.mask_model, streaming=args.stream)
    }[args.controller]

    controller = controller_cls(car)

    print(f"Using {args.controller} controller.")
    controller.run()

if __name__ == "__main__":
    main()
