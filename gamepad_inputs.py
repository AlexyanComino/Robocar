##
## EPITECH PROJECT, 2025
## RobocarProject
## File description:
## inputs
##

from inputs import devices

class GamepadInput:
    """
    Class to handle gamepad inputs.
    This class provides methods to read the state of the gamepad.
    """

    def __init__(self):
        self.gamepad_state = {}

    def update(self):
        """Update the gamepad state by reading the current inputs."""
        events = devices.gamepads[0]._do_iter()
        if not events:
            return []
        updated = []
        for event in events:
            if event.ev_type in ('Key', 'Absolute'):
                prev_state = self.gamepad_state.get(event.code, 0)
                self.gamepad_state[event.code] = event.state
                if prev_state != event.state:
                    updated.append((event.code, event.state))
        return updated

    def get_state(self, code):
        """Get the current state of a specific gamepad input."""
        return self.gamepad_state.get(code, 0)

def main():
    gamepad_input = GamepadInput()

    # Main loop to continuously update and print gamepad state
    try:
        while True:
            updated = gamepad_input.update()
            for code, state in updated:
                print(f"{code}: {state}")
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()