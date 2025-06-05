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

def run(controller, car):
    """
    Run the main loop of the Robocar project.

    Args:
        controller (IController): The controller to use for handling inputs.
    """
    while True:
        updated = controller.update()
        if updated:
            actions = controller.get_actions()
            car.set_actions(actions)


def main():
    parser = ArgumentParser(description="Robocar Project")
    parser.add_argument(
        "--controller",
        choices=["gamepad", "ai"],
        default="gamepad",
        help="Select the controller type: 'gamepad' or 'ai'. Default is 'gamepad'."
    )

    args = parser.parse_args()
    if args.controller == "gamepad":
        controller = GamepadController()
    elif args.controller == "ai":
        input_size = 57
        hidden_layers = [32, 64, 128, 64, 32]
        output_size = 2
        controller = AIController(input_size, hidden_layers, output_size)
    else:
        raise ValueError("Invalid controller type. Choose 'gamepad' or 'ai'.")

    print(f"Using {args.controller} controller.")

    car = Car(port=PORT, power_limit=0.1)

    run(controller, car)

if __name__ == "__main__":
    main()
