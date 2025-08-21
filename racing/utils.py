##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## utils
##

import torch
import os

from racing.model import MyModel

from logger import setup_logger

logger = setup_logger(__name__)

def load_racing_model(racing_model_path: str, input_columns: list, hidden_layers: list, output_size: int) -> MyModel:
    """
    Load the Racing model from the specified path.
    Args:
        racing_model_path: Path to the Racing model file.
    Returns:
        Loaded Racing model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyModel(input_size=len(input_columns), hidden_layers=hidden_layers, output_size=output_size).to(device)

    if os.path.exists(racing_model_path.replace('.pth', '.pt')):
        model = torch.jit.load(racing_model_path.replace('.pth', '.pt'), map_location=device)
        model.eval()
    else:
        logger.info(f"Traced Racing model not found, creating a new one.")
        model.eval()
        example_input = torch.randn(len(input_columns), device=device)
        model = torch.jit.trace(model, example_input)
        torch.jit.save(model, racing_model_path.replace('.pth', '.pt'))
        logger.info(f"Racing model saved to {racing_model_path.replace('.pth', '.pt')}")

    return model
