##
## EPITECH PROJECT, 2025
## robocar [SSH: robocar-matys]
## File description:
## trt_inference
##

import os
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TRTInference:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")

        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Assume one input, one output
        self.input_binding_idx = self.engine.get_binding_index(self.engine[0])
        self.output_binding_idx = self.engine.get_binding_index(self.engine[1])

        # Dynamic shape support
        self.input_shape = self.engine.get_binding_shape(self.input_binding_idx)
        self.dtype = trt.nptype(self.engine.get_binding_dtype(self.input_binding_idx))

    def infer(self, input_tensor: torch.Tensor) -> np.ndarray:
        assert input_tensor.is_cuda, "Input tensor must be on CUDA device"
        batch_size, channels, height, width = input_tensor.shape

        # Set input shape for dynamic dimensions
        self.context.set_binding_shape(self.input_binding_idx, input_tensor.shape)

        # Output info
        output_shape = self.context.get_binding_shape(self.output_binding_idx)
        output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_binding_idx))
        output_size = np.prod(output_shape) * np.dtype(output_dtype).itemsize

        # Output Allocation
        d_output = cuda.mem_alloc(output_size)

        d_input_ptr = input_tensor.data_ptr()
        bindings = [int(d_input_ptr), int(d_output)]

        # Run inference
        self.context.execute_v2(bindings)

        # Transfer predictions back
        output = np.empty(output_shape, dtype=output_dtype)
        cuda.memcpy_dtoh(output, d_output)

        return output
