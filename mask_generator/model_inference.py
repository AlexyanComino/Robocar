##
## EPITECH PROJECT, 2025
## RobocarProject
## File description:
## model_loader
##

import os
import cv2
import torch
import random
import numpy as np
import sys
import os
import yaml
import torch.nn as nn
from mask_generator.config import ModelConfig
from mask_generator.models.utils import create_model
from mask_generator.transforms import EvalTransform, TensorDecoder

def load_model_config_from_yaml(yaml_path: str) -> ModelConfig:
    """Load model configuration from a YAML file."""
    with open(yaml_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return ModelConfig(**config_data.get('model', {}))

def load_model_from_config(config_path: str, checkpoint_path: str) -> nn.Module:
    """Load a model from a configuration file and a checkpoint."""
    config = load_model_config_from_yaml(config_path)
    model, pad_divisor = create_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model to device: {device}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, pad_divisor

def load_model_from_run_dir(run_dir: str) -> nn.Module:
    """Load a model from a run directory containing config and checkpoint files."""
    config_path = os.path.join(run_dir, 'config.yaml')
    checkpoint_path = os.path.join(run_dir, 'model.pth')
    if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Configuration or checkpoint file not found in the run directory.")
    return load_model_from_config(config_path, checkpoint_path)


def infer_mask(model: nn.Module, pad_divisor: int, image: np.ndarray) -> np.ndarray:
    """Run inference on an image to generate a mask."""
    transform = EvalTransform(pad_divisor=pad_divisor, to_tensor=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tensor_decoder = TensorDecoder()

    # Preprocess the image
    input_tensor = transform(image=image).unsqueeze(0)

    # Move to device
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)

    # Decode the output tensor to a mask
    mask_np = decoder.to_mask(output.cpu())
    return mask_np
