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

    def get_speed(self):
        """
        Get the current speed of the car.

        :return: The current speed in m/s.
        """
        rpm = self.vesc.get_rpm()
        if rpm is None:
            return 0.0
        wheel_diameter = 0.01  # Example wheel diameter in meters
        wheel_circumference = 3.14159 * wheel_diameter
        speed = (rpm * wheel_circumference) / 60.0  # Convert RPM to m/s
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
        duty_cycle *= self.power_limit
        if duty_cycle < -self.power_limit:
            duty_cycle = -self.power_limit
        elif duty_cycle > self.power_limit:
            duty_cycle = self.power_limit
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
