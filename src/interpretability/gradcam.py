from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.models.ct import IMAGENET_MEAN, IMAGENET_STD


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    image = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN)
    return np.clip(image, 0, 1)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module: torch.nn.Module, grad_input: Any, grad_output: Any) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None, size: tuple[int, int] = (224, 224)) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()
        return (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
