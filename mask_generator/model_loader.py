##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## model_loader
##

import os
import cv2
import sys
import os
import yaml
import torch
import torch.nn as nn
from mask_generator.config import ModelConfig
from mask_generator.models.utils import create_model

def load_model_config_from_yaml(yaml_path: str) -> ModelConfig:
    """Load model configuration from a YAML file."""
    with open(yaml_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return ModelConfig(**config_data.get('model', {}))

def load_model_from_config(config_path: str, checkpoint_path: str, device: str) -> nn.Module:
    """Load a model from a configuration file and a checkpoint."""
    config = load_model_config_from_yaml(config_path)
    model, pad_divisor = create_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, pad_divisor

def load_model_from_run_dir(run_dir: str, device: str) -> nn.Module:
    """Load a model from a run directory containing config and checkpoint files."""
    config_path = os.path.join(run_dir, 'config.yaml')
    checkpoint_path = os.path.join(run_dir, 'model.pth')
    if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Configuration or checkpoint file not found in the run directory.")
    return load_model_from_config(config_path, checkpoint_path, device)
