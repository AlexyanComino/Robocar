##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## icontroller
##

from abc import ABC, abstractmethod

class IController(ABC):
    """
    Interface for all controllers in the Robocar project.
    """

    @abstractmethod
    def run(self, car):
        """
        Run the controller with the given car instance.
        This method should be implemented by all controllers.

        Args:
            car: An instance of the Car class to control.
        """
        pass
