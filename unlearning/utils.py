"""Utility helpers shared across unlearning modules."""

from __future__ import annotations

from typing import Dict

import torch
from torchvision import models


def instantiate_model(state_dict: Dict[str, torch.Tensor], num_classes: int = 10) -> torch.nn.Module:
    """Instantiate a ResNet-18 model and load a (possibly nested) state dict."""

    model = models.resnet18(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        for key in ("model_state_dict", "net_sd", "state_dict"):
            if key in state_dict:
                missing, unexpected = model.load_state_dict(state_dict[key], strict=False)
                break
    return model
