"""Class‑discriminative pruning for machine unlearning.

This module implements a simplified variant of class‑discriminative
pruning for federated unlearning (Yu et al., 2021).  Channels in the
first convolutional layer that are most indicative of the forgotten
class are zeroed out, and the network is fine‑tuned on the remaining
data.  Only the absolute weight magnitudes are used as a proxy for
class discriminativity in this example.  A single fine‑tuning batch
keeps the computation lightweight.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .base import UnlearningMethod
from .registry import register_mu


@register_mu("prune")
class PruneClassUnlearning(UnlearningMethod):
    """Minimal class‑discriminative pruning unlearning implementation."""

    def run(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        model = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        conv_layer = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                conv_layer = m
                break
        if conv_layer is None:
            raise RuntimeError("No convolutional layer found in the model; cannot prune.")
        weights = conv_layer.weight.data.detach().abs().mean(dim=(1, 2, 3))
        prune_ratio = float(getattr(self.args, "prune_ratio", 0.2))
        num_channels = weights.numel()
        num_prune = max(1, int(prune_ratio * num_channels))
        _, prune_indices = torch.topk(weights, num_prune, largest=True)
        mask = torch.ones(num_channels, dtype=torch.bool)
        mask[prune_indices] = False
        with torch.no_grad():
            conv_layer.weight.data[prune_indices] = 0.0
            if conv_layer.bias is not None:
                conv_layer.bias.data[prune_indices] = 0.0
        self.mask = mask.clone().detach().cpu()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.train()
        for x, y in self.train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            break
        model.cpu()
        metrics = {
            "pruned_channels": int(num_prune),
            "prune_ratio": prune_ratio,
        }
        return model, metrics
