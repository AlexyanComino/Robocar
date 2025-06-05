##
## EPITECH PROJECT, 2025
## RobocarProject
## File description:
## model_loader
##

import torch
import numpy as np
import torch.nn as nn
from mask_generator.transforms import EvalTransform, TensorDecoder

def infer_mask(model: nn.Module, transform: EvalTransform, decoder: TensorDecoder, image: np.ndarray, device: str) -> np.ndarray:
    """Run inference on an image to generate a mask."""
    # Preprocess the image
    input_tensor = transform(image=image).unsqueeze(0).to(device, non_blocking=True)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)

    decoder = TensorDecoder()

    # Decode the output tensor to a mask
    mask_np = decoder.to_mask(output.cpu())
    return mask_np
