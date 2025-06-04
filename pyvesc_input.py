from pyvesc import VESC
from gamepad_inputs import GamepadInput

PORT = "/dev/ttyACM0"

def set_servo_position(vesc, position):
    """
    Set the servo position on the VESC.

    :param vesc: The VESC instance to send the command to.
    :param position: The desired servo position (0.0 to 1.0).
    """
    if position < 0.0:
        position = 0.0
    elif position > 1.0:
        position = 1.0
    vesc.set_servo(position)


def set_duty_cycle(vesc, duty_cycle):
    """
    Set the duty cycle on the VESC.

    :param vesc: The VESC instance to send the command to.
    :param duty_cycle: The desired duty cycle (-1.0 to 1.0).
    """
    limit = 0.3
    duty_cycle *= limit
    if duty_cycle < -limit:
        duty_cycle = -limit
    elif duty_cycle > limit:
        duty_cycle = limit
    vesc.set_duty_cycle(duty_cycle)


def handle_events(updated, vesc):
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
            set_servo_position(vesc, position)
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
            set_duty_cycle(vesc, -state / 255.0)
        elif code == 'ABS_RZ':
            print(f"Right Trigger Axis: {state}")
            set_duty_cycle(vesc, state / 255.0)

def main():
    gamepad_input = GamepadInput()
    vesc = VESC(serial_port=PORT)

    try:
        while True:
            updated = gamepad_input.update()
            handle_events(updated, vesc)
    except KeyboardInterrupt:
        print("Exiting...")

    # vesc.set_duty_cycle(0)
    # vesc.set_servo_position(0)

if __name__ == "__main__":
    main()
