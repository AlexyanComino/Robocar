##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## model
##

import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        """
        A generic feedforward neural network

        Args:
            input_size (int): The size of the input
            hidden_layers (list): The size of the hidden layers
            output_size (int): The size of the output
        """
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        A generic LSTM neural network

        Args:
            input_size (int): The size of the input
            hidden_size (int): The size of the hidden layers
            output_size (int): The size of the output
            num_layers (int): The number of LSTM layers
        """
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
