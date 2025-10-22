"""Shard‑based machine unlearning (SISA).

This implementation provides a minimal working version of the SISA
algorithm described by Bourtoule et al. (2021).  The idea is to
partition the dataset into several disjoint shards and fine‑tune a
separate copy of the model on each shard (after removing the
forgotten samples).  The final unlearned model is obtained by
averaging the parameters of all shard models.  Only a single mini‑
batch is used for fine‑tuning each shard to keep this example
lightweight.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .base import UnlearningMethod
from .registry import register_mu
from .utils import instantiate_model


@register_mu("sisa")
class SISAUnlearning(UnlearningMethod):
    """Minimal SISA unlearning implementation."""

    def run(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        base_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        dataset = self.train_loader.dataset
        num_shards = max(1, int(getattr(self.args, "sisa_shards", 8)))
        indices = np.arange(len(dataset))
        shards = np.array_split(indices, num_shards)
        states = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for shard_indices in shards:
            model_i = instantiate_model(base_state, num_classes=10)
            model_i.load_state_dict(base_state)
            model_i.to(device)
            dl = DataLoader(Subset(dataset, shard_indices.tolist()),
                            batch_size=getattr(self.args, "batch_size", 64),
                            shuffle=True, num_workers=0)
            optimizer = torch.optim.SGD(model_i.parameters(), lr=0.01)
            model_i.train()
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model_i(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
                break
            states.append({k: v.detach().cpu().clone() for k, v in model_i.state_dict().items()})
        avg_state: Dict[str, torch.Tensor] = {}
        for name in base_state:
            stacked = torch.stack([s[name] for s in states], dim=0)
            avg_state[name] = stacked.mean(dim=0)
        new_model = instantiate_model(base_state, num_classes=10)
        new_model.load_state_dict(avg_state)
        metrics = {
            "shards": num_shards,
            "steps_per_shard": 1,
        }
        return new_model, metrics
