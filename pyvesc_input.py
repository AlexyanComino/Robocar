import serial
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
    limit = 0.1
    if duty_cycle < -limit:
        duty_cycle = -limit
    elif duty_cycle > limit:
        duty_cycle = limit
    vesc.set_duty_cycle(duty_cycle)


def main():
    gamepad_input = GamepadInput()
    vesc = VESC(serial_port=PORT)
    # vesc.set_duty_cycle(0.01)

    # Main loop to continuously update and print gamepad state
    try:
        while True:
            updated = gamepad_input.update()
            for code, state in updated:
                if code == 'ABS_Y':
                    # Assuming ABS_Y is the left joystick vertical axis
                    duty_cycle = -state / 32767.0
                    duty_cycle = duty_cycle * 0.1
                    print(f"Setting duty cycle to: {duty_cycle}")
                    set_duty_cycle(vesc, duty_cycle)
                elif code == 'ABS_X':
                    # Assuming ABS_X is the left joystick horizontal axis
                    servo_position = (state + 32767) / 65534.0
                    servo_position = max(0.0, min(1.0, servo_position))
                    print(f"Setting servo position to: {servo_position}")
                    set_servo_position(vesc, servo_position)
    except KeyboardInterrupt:
        print("Exiting...")

    vesc.set_duty_cycle(0)
    vesc.set_servo_position(0)

if __name__ == "__main__":
    main()
