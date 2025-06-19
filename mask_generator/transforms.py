##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## transform_manager
##

import numpy as np
import kornia.geometry.transform as kt
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        raise NotImplementedError("Subclasses should implement this method.")

class KorniaInferTransform(BaseTransform):
    def __init__(self, pad_divisor: int, target_height: int = 256, device: str = 'cpu'):
        self.pad_divisor = pad_divisor
        self.target_height = target_height
        self.device = device

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

    def _resize_with_aspect_ratio(self, img: torch.Tensor, interpolation: str = 'bilinear') -> torch.Tensor:
        if interpolation not in ['bilinear', 'nearest']:
            raise ValueError("Interpolation must be either 'bilinear' or 'nearest'.")

        _, h, w = img.shape
        new_w = int(self.target_height * w / h) if h != 0 else w
        img = kt.resize(img, (self.target_height, new_w), interpolation=interpolation)
        return img

    def _pad_if_needed(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        pad_h = (self.pad_divisor - h % self.pad_divisor) % self.pad_divisor
        pad_w = (self.pad_divisor - w % self.pad_divisor) % self.pad_divisor
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            padding = (pad_left, pad_right, pad_top, pad_bottom)
            img = F.pad(img, padding, mode='constant', value=0)
        return img

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize the image tensor."""
        return (img - self.mean) / self.std

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        assert isinstance(image, np.ndarray), "Image must be a numpy array."
        assert image.ndim == 3 and image.shape[2] == 3, "Image must be HWC with 3 channels"

        img_tensor = torch.from_numpy(image).to(self.device).permute(2, 0, 1).float() / 255.0 # [H, W, C] -> [C, H, W]
        img_tensor = self._resize_with_aspect_ratio(img_tensor, interpolation='bilinear')
        img_tensor = self._pad_if_needed(img_tensor)
        img_tensor = self._normalize(img_tensor)

        if mask is not None:
            assert isinstance(mask, np.ndarray), "Mask must be a numpy array."
            assert mask.ndim == 2, "Mask must be grayscale with shape [H, W]"

            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float().to(self.device) # [H, W] -> [1, H, W]
            mask_tensor = self._resize_with_aspect_ratio(mask_tensor, interpolation='nearest')
            mask_tensor = self._pad_if_needed(mask_tensor)
            return img_tensor, mask_tensor
        return img_tensor

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse normalization of a tensor."""
        return tensor * self.std + self.mean

    def to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor to a numpy image."""

        if tensor.dim() == 4:
            tensor = tensor.squeeze(0) # [B, C, H, W] -> [C, H, W]
        image = self.denormalize(tensor).clamp(0, 1)
        image = image.permute(1, 2, 0).cpu().numpy() # [C,H,W] -> [H,W,C]
        return (image * 255).astype(np.uint8)

    def to_mask(self, tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0) # [B, C, H, W] -> [C, H, W]
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0) # [C, H, W] -> [H, W]

        mask = tensor.cpu().numpy()
        return (mask > threshold).astype(np.uint8) * 255

    # def to_mask_tensor(self, tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    #     if tensor.dim() == 4:
    #         tensor = tensor.squeeze(0) # [B, C, H, W] -> [C, H, W]
    #     if tensor.dim() == 3:
    #         tensor = tensor.squeeze(0) # [C, H, W] -> [H, W]

    #     return (tensor > threshold).float() * 255.0
