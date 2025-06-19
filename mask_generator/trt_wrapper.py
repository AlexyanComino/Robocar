##
## EPITECH PROJECT, 2025
## robocar [SSH: robocar-matys]
## File description:
## trt_wrapper
##

import os
import time
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from logger import setup_logger, TimeLogger

logger = setup_logger(__name__)

class TRTWrapper:
    def __init__(self, engine_path: str, device: str = "cuda"):
        self.device = device
        print(f"Using device: {self.device}")
        print(self.device == "cuda")
        print(device)
        assert device != "cuda", "TensorRT only supports CUDA device"
        trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(trt_logger)

        if not os.path.exists(engine_path):
            logger.error(f"TensorRT engine file not found: {engine_path}")
            raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")
        if not os.path.isfile(engine_path):
            logger.error(f"TensorRT engine file does not exist: {engine_path}")
            raise FileNotFoundError(f"TensorRT engine file does not exist: {engine_path}")

        with TimeLogger(f"Reading TensorRT engine from {engine_path}", logger):
            with open(engine_path, 'rb') as f:
                engine_data = f.read()

        with TimeLogger("Deserializing TensorRT engine", logger):
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)

        with TimeLogger("Creating TensorRT execution context", logger):
            self.context = self.engine.create_execution_context()

        # Bindings
        self.bindings = [None] * self.engine.num_bindings
        self.input_binding_idx = []
        self.output_binding_idx = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                self.input_binding_idx.append(i)
                logger.debug(f"Input binding found: {name} (index {i})")
            else:
                self.output_binding_idx.append(i)
                logger.debug(f"Output binding found: {name} (index {i})")

        assert len(self.input_binding_idx) == 1, "Expected exactly one input"
        assert len(self.output_binding_idx) == 1, "Expected exactly one output"

        self.input_index = self.input_binding_idx[0]
        self.output_index = self.output_binding_idx[0]
        self.input_name = self.engine.get_binding_name(self.input_index)
        self.output_name = self.engine.get_binding_name(self.output_index)

        logger.info(f"Input : {self.input_name} (index {self.input_index})")
        logger.info(f"Output : {self.output_name} (index {self.output_index})")

        # Dynamic shape support
        self.input_shape = self.engine.get_binding_shape(self.input_index)
        logger.info(f"Input shape dynamique : {self.input_shape}")

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.forward(input_tensor)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        assert input_tensor.is_cuda, "Input tensor must be on CUDA device"

        with TimeLogger("Preparing TensorRT bindings", logger):
            # Set input shape for dynamic dimensions
            self.context.set_binding_shape(self.input_index, input_tensor.shape)

            # Output info
            output_shape = self.context.get_binding_shape(self.output_index)
            output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_index))
            torch_dtype = torch.from_numpy(np.empty((), dtype=output_dtype)).dtype

            # Allocate output tensor directly on CUDA
            output_tensor = torch.empty(size=output_shape, dtype=torch_dtype, device=self.device)

            # Set bindings using raw data pointers
            self.bindings[self.input_index] = int(input_tensor.data_ptr())
            self.bindings[self.output_index] = int(output_tensor.data_ptr())

        # Run inference
        with TimeLogger("Inference TensorRT", logger=logger):
            self.context.execute_v2(self.bindings)

        return output_tensor
