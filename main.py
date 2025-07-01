##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## main
##

import sys
import argparse

from controllers.gamepad_controller import GamepadController
from controllers.gamepad_writer_controller import GamepadWriterController
from controllers.ai_controller import AIController
from car import Car
from logger import setup_logger

logger = setup_logger(__name__)

PORT = "/dev/ttyACM0"

def parse_args():
    parser = argparse.ArgumentParser(description="Robocar Project")

    subparsers = parser.add_subparsers(dest="controller", help="Controller type")

    gamepad_parser = subparsers.add_parser("gamepad", help="Control the car with a gamepad")

    gamepad_writer_parser = subparsers.add_parser("gamepadW", help="Control the car with a gamepad and write data to CSV")

    gamepad_writer_parser.add_argument(
        "--mask-model",
        type=str,
        default="mask_generator/20250626_200538_96718eef57",
        help="Path to the mask generator model directory."
    )

    gamepad_writer_parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable camera stream for gamepad writer controller."
    )

    ai_parser = subparsers.add_parser("ai", help="Control the car with AI model")
    ai_parser.add_argument(
        "--mask-model",
        type=str,
        default="mask_generator/20250626_200538_96718eef57",
        help="Path to the mask generator model directory."
    )
    ai_parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable camera stream for AI controller."
    )

    args = parser.parse_args()

    if not args.controller:
        parser.print_help()
        sys.exit(1)

    return args


def main():
    logger.info("Starting Robocar main script")
    args = parse_args()

    car = Car(port=PORT, power_limit=0.04)
    controller_cls = {
        "gamepad": GamepadController,
        "ai": lambda car: AIController(car, mask_model_dir=args.mask_model, streaming=args.stream),
        "gamepadW": lambda car: GamepadWriterController(car, mask_model_dir=args.mask_model, streaming=args.stream)
    }[args.controller]

    controller = controller_cls(car)

    logger.info(f"Using {args.controller} controller.")
    controller.run()

if __name__ == "__main__":
    main()
