##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## utils
##

import time
import torch
import numpy as np
from mask_generator.transforms import KorniaInferTransform
from mask_generator.trt_inference import TRTInference

def infer_mask(trt_infer: TRTInference, transform: KorniaInferTransform, image: np.ndarray) -> np.ndarray:
    """Run inference on an image to generate a mask."""

    # print("Transforming image")
    # t1 = time.time()
    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0).contiguous()
    # print("Is contiguous:", img_tensor.is_contiguous())
    # print(f"[INFER_MASK] Transform image exectuted in {time.time() - t1:.4f}s")

    output = trt_infer.infer(img_tensor)
    output = torch.sigmoid(output)

    mask_np = transform.to_mask(output.cpu())
    return mask_np
