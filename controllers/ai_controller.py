##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## ai_controller
##

import torch
import pickle
import joblib

from controllers.icontroller import IController
from AI.model import MyModel

class AIController(IController):
    """
    Controller for the AI model in the Robocar project.
    This controller handles the interaction with the AI model.
    """

    def __init__(self, input_size, hidden_layers, output_size):
        """
        Initialize the AIController with a model.

        Args:
            input_size (int): The size of the input layer.
            hidden_layers (list): The sizes of the hidden layers.
            output_size (int): The size of the output layer.
        """
        def load_model(input_size, hidden_layers, output_size):
            model = MyModel(input_size, hidden_layers, output_size)

            model_path = "model24220ce995.joblib"

            save_dict = joblib.load(model_path)
            model.load_state_dict(save_dict["model_weights"])
            scaler = save_dict["scaler"]

            model.eval()
            return model, scaler

        self.model, self.scaler = load_model(input_size, hidden_layers, output_size)

    def predict(self, input_data):
        """
        Make a prediction using the AI model.

        Args:
            input_data (torch.Tensor): The input data for prediction.

        Returns:
            torch.Tensor: The model's prediction.
        """
        with torch.no_grad():
            prediction = self.model(input_data).numpy().squeeze()
        return prediction

    def update(self):
        """
        Update the AI controller state.
        This method should be implemented by all controllers.
        """
        # In a real implementation, this might involve processing input data
        # and updating the model state. Here, we just return an empty list.
        return True

    def get_actions(self):
        """
        Get the actions to be performed by the car based on the AI model's prediction.

        Returns:
            dict: A dictionary containing the actions derived from the AI model.
        """

        # Add mask generator logic and input data preparation here
        input_data = torch.zeros((1, 57))
        input_data = self.scaler.transform(input_data.numpy())
        data_scaled = self.scaler.transform([input_data])
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32)


        prediction = self.predict(data_tensor)
        print(f"Prediction: {prediction}")
        return {
            "throttle": prediction[0],
            "steering": prediction[1]
        }
