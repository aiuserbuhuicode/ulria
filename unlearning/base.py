"""Abstract base class for machine unlearning algorithms.

This module defines the common interface for machine unlearning
algorithms implemented in this repository. Each algorithm should
inherit from :class:`UnlearningMethod` and implement the :meth:`run`
method to return a new model after unlearning along with a metrics
dictionary. See the accompanying paper and referenced repositories
for more details on the underlying techniques.

Sources:
    - SISA (Shard-based incremental selection algorithms) [cleverhans-lab/machine-unlearning]
    - Selective Forgetting via Fisher Information (Golatkar et al., 2019) [arXiv:1911.04933]
    - Knowledge Distillation based unlearning (teacher–student frameworks) [jjbrophy47/machine_unlearning]
    - Class-discriminative pruning for federated unlearning (Yu et al., 2021) [arXiv:2110.11794]
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Tuple

import torch


class UnlearningMethod(abc.ABC):
    """Abstract base class defining the interface for unlearning methods."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        forget_spec: Tuple[str, Any],
        args: Any,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.forget_spec = forget_spec
        self.args = args

    @abc.abstractmethod
    def run(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Perform the unlearning procedure and return the new model and metrics."""
        raise NotImplementedError
