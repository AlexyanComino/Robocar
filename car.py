##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## car
##

from pyvesc import VESC

MAX_LIMIT = 0.3

class Car:
    def __init__(self, port, power_limit=MAX_LIMIT):
        """
        Initialize the Car instance.

        :param vesc: The VESC instance to control the car.
        :param power_limit: The maximum power limit for the car.
        """
        self.vesc = VESC(serial_port=port)
        self.power_limit = power_limit
        self.old_speed = 0.0

    def __del__(self):
        """
        Clean up the Car instance by closing the VESC connection.
        """
        self.set_duty_cycle(0.0)  # Ensure motors are stopped
        self.set_servo(0.5)  # Center the servo

    def get_vesc(self):
        """
        Get the VESC instance.

        :return: The VESC instance.
        """
        return self.vesc

    def get_old_speed(self):
        """
        Get the last known speed of the car.

        :return: The last known speed in m/s.
        """
        return self.old_speed

    def get_speed(self):
        """
        Get the current speed of the car.

        :return: The current speed in m/s.
        """
        try:
            rpm = self.vesc.get_rpm()
        except AttributeError:
            return self.old_speed

        if rpm is None:
            return self.old_speed
        wheel_radius = 0.005  # Example wheel radius in meters
        wheel_circumference = 2 * 3.1415926535 * wheel_radius
        speed = (wheel_circumference * rpm) / 60.0
        self.old_speed = speed
        return speed

    def get_power_limit(self):
        """
        Get the current power limit for the car.

        :return: The current power limit.
        """
        return self.power_limit

    def set_power_limit(self, power_limit):
        """
        Set the power limit for the car.

        :param power_limit: The new power limit.
        """
        self.power_limit = power_limit

    def set_duty_cycle(self, duty_cycle):
        """
        Set the duty cycle on the VESC.

        :param duty_cycle: The desired duty cycle (-1.0 to 1.0).
        """
        limit = self.power_limit or 0.01
        duty_cycle *= limit
        if duty_cycle < -limit:
            duty_cycle = -limit
        elif duty_cycle > limit:
            duty_cycle = limit
        self.vesc.set_duty_cycle(duty_cycle)

    def set_servo(self, position):
        """
        Set the servo position on the VESC.

        :param position: The desired servo position (0.0 to 1.0).
        """
        if position < 0.0:
            position = 0.0
        elif position > 1.0:
            position = 1.0
        self.vesc.set_servo(position)

    def increment_power_limit(self, increment):
        """
        Increment the power limit by a specified amount.

        :param increment: The amount to increment the power limit.
        """
        new_limit = self.power_limit + increment
        if new_limit < 0.01:
            new_limit = 0.01
        if new_limit > MAX_LIMIT:
            new_limit = MAX_LIMIT
        self.set_power_limit(new_limit)

    def set_actions(self, actions: dict):
        """
        Set the actions to be performed by the car.

        :param actions: A dictionary containing 'throttle' and 'steering' values.
        """
        self.set_duty_cycle(actions['throttle'])
        self.set_servo(actions['steering'])
