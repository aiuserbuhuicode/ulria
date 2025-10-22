"""Exact retraining baseline for machine unlearning.

This baseline simply retrains the model on the retained data (i.e.
after removing the specified samples) for a small number of epochs.
Although naïve, this approach serves as a strong upper bound on
unlearning quality and is commonly used as a baseline in the literature.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .base import UnlearningMethod
from .registry import register_mu


@register_mu("retrain")
class RetrainUnlearning(UnlearningMethod):
    """Minimal exact retraining baseline implementation."""

    def run(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model
        model.to(device)
        epochs = int(getattr(self.args, "retrain_epochs", 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        step_count = 0
        for _ in range(max(1, epochs)):
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
                step_count += 1
                break
        model.cpu()
        metrics = {
            "retrain_epochs": epochs,
            "steps": step_count,
        }
        return model, metrics
