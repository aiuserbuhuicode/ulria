"""Selective forgetting via Fisher information.

This module implements a minimal version of the selective forgetting
approach proposed by Golatkar et al. (2019).  The diagonal of the
Fisher information matrix (FIM) is estimated as an exponential moving
average of per‑sample squared gradients.  Parameters aligned with
high‑Fisher directions are regularised and perturbed before a short
fine‑tuning.  This simplified implementation uses only a few mini‑
batches to estimate the FIM and a single batch for fine‑tuning.

Reference: Aditya Golatkar, Alessandro Achille, C. Studer, and Stefano
Soatto. “Eternal Sunshine of the Spotless Net: Selective Forgetting in
Deep Networks”. arXiv:1911.04933, 2019.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .base import UnlearningMethod
from .registry import register_mu


@register_mu("fisher")
class FisherUnlearning(UnlearningMethod):
    """Minimal Fisher information based unlearning implementation."""

    def run(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model
        model.to(device)
        # Estimate diagonal FIM
        fisher_batches = int(getattr(self.args, "fisher_batches", 64))
        fim: Dict[str, torch.Tensor] = {name: torch.zeros_like(param, device=device)
                                        for name, param in model.named_parameters()
                                        if param.requires_grad}
        batch_count = 0
        for i, (x, y) in enumerate(self.train_loader):
            if i >= fisher_batches:
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            grads = torch.autograd.grad(loss,
                                        [p for p in model.parameters() if p.requires_grad],
                                        retain_graph=False)
            # Accumulate squared gradients
            for (name, param), g in zip(model.named_parameters(), grads):
                if param.requires_grad:
                    fim[name] += (g.detach() ** 2)
            batch_count += 1
        if batch_count == 0:
            batch_count = 1
        for name in fim:
            fim[name] = fim[name] / float(batch_count)
        # Apply Fisher scaling and noise
        scale = float(getattr(self.args, "fisher_scale", 1e-3))
        noise_std = float(getattr(self.args, "fisher_noise_std", 0.0))
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in fim:
                    param.add_(-scale * fim[name] * param)
                    if noise_std > 0.0:
                        param.add_(torch.randn_like(param) * noise_std)
        # Short fine‑tuning on the reduced dataset (one batch)
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
            "fisher_batches": fisher_batches,
        }
        return model, metrics
