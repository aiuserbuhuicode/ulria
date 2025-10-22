"""Knowledge distillation based machine unlearning.

This module provides a minimal implementation of teacher–student
distillation for machine unlearning.  The teacher is a frozen copy of
the original model and guides the student (a mutable copy of the
original) to forget the specified samples.  The loss combines
temperature‑scaled KL divergence between teacher and student outputs
with the standard cross‑entropy on the retained data.  Only a single
mini‑batch per epoch is used to keep the example lightweight.

Representative works on distillation‑based unlearning can be found in
the survey at https://github.com/jjbrophy47/machine_unlearning.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .base import UnlearningMethod
from .registry import register_mu


@register_mu("kd")
class KDUnlearning(UnlearningMethod):
    """Minimal teacher–student unlearning implementation."""

    def run(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        teacher = copy.deepcopy(self.model)
        teacher.eval()
        student = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher.to(device)
        student.to(device)
        T = float(getattr(self.args, "kd_T", 2.0))
        alpha = float(getattr(self.args, "kd_alpha", 0.5))
        epochs = int(getattr(self.args, "kd_epochs", 1))
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        for _ in range(max(1, epochs)):
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    logits_t = teacher(x)
                logits_s = student(x)
                loss_kd = F.kl_div(
                    F.log_softmax(logits_s / T, dim=1),
                    F.softmax(logits_t / T, dim=1),
                    reduction="batchmean",
                ) * (T * T)
                loss_ce = F.cross_entropy(logits_s, y)
                loss = alpha * loss_kd + (1.0 - alpha) * loss_ce
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break  # one batch per epoch
        student.cpu()
        metrics = {
            "kd_T": T,
            "kd_alpha": alpha,
            "kd_epochs": epochs,
        }
        return student, metrics
