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
    def update(self):
        """
        Update the controller state.
        This method should be implemented by all controllers.
        """
        pass

    @abstractmethod
    def get_actions(self) -> dict:
        """
        Get the actions to be performed by the car.
        This method should be implemented by all controllers.
        """
        pass
