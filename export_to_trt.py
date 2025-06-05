##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## export_to_trt
##

import torch
from torch2trt import torch2trt
from mask_generator.model_loader import load_model_from_run_dir
import os

MODEL_RUN_DIR = "mask_generator/best_run"
OUTPUT_PATH = "mask_generator/best_run/model_trt.pth"

def convert_to_trt(model, input_shape):
    """
    Convert a PyTorch model to TensorRT format.

    :param model: The PyTorch model to convert.
    :param input_shape: The shape of the input tensor (batch_size, channels, height, width).
    """
    # Create a dummy input tensor with the specified shape
    dummy_input = torch.randn(input_shape).cuda()

    # Convert the model to TensorRT
    model_trt = torch2trt(model,
        [dummy_input],
        fp16_mode=True,
        log_level=torch2trt.logging.Level.VERBOSE
    )

def save_trt_model(model_trt, output_path):
    torch.save(model_trt.state_dict(), output_path)
    print(f"Model saved to {output_path}")

def main():
    assert torch.cuda.is_available(), "CUDA is not available. Please run this script on a machine with a CUDA-enabled GPU."

    model = load_model_from_run_dir(MODEL_RUN_DIR)
    model.eval()
    model.cuda()
    input_shape = (1, 3, 256, 455)  # Example input shape (batch_size, channels, height, width)
    model_trt = convert_to_trt(model, input_shape)
    save_trt_model(model_trt, OUTPUT_PATH)

if __name__ == "__main__":
    main()