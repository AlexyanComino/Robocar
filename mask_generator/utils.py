##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## utils
##

import torch
import numpy as np
from mask_generator.transforms import KorniaInferTransform
from mask_generator.trt_wrapper import TRTWrapper
from logger import setup_logger, TimeLogger

logger = setup_logger(__name__)

def get_mask(mask_model: TRTWrapper, transform: KorniaInferTransform, image: np.ndarray) -> torch.Tensor:
    """Run inference on an image to generate a mask."""

    # Preprocess the image
    with TimeLogger("Transforming image", logger):
        img_tensor = transform(image).unsqueeze(0).contiguous()

    with TimeLogger("Running inference on mask model", logger):
        output = mask_model(img_tensor)
        output = torch.sigmoid(output)

    with TimeLogger("Converting output to mask", logger):
        mask_np = transform.to_mask(output.cpu())
    return mask_np

