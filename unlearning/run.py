"""Unified command-line interface for machine unlearning baselines.

This entry point exposes a simple interface to perform a variety of
machine unlearning baselines on a given pretrained model.  It
supports exact retraining, shard-based (SISA) unlearning, Fisher
information-based selective forgetting, knowledge distillation based
unlearning, and class‑discriminative pruning.  Additional methods can
be registered in :mod:`unlearning.registry`.  To run this module
directly invoke it via ``python -m unlearning.run``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .registry import REGISTRY
from .utils import instantiate_model


def parse_forget_spec(spec: str, dataset: datasets.VisionDataset) -> Tuple[str, Any, List[int]]:
    """Interpret the ``--forget-spec`` argument."""
    kind: str
    value: Any
    forget_indices: List[int] = []
    if spec.startswith("class="):
        kind = "class"
        value = int(spec.split("=", 1)[1])
        forget_indices = [i for i, (_, y) in enumerate(dataset) if int(y) == value]
    else:
        if os.path.isfile(spec):
            with open(spec, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "indices" in data:
                forget_indices = list(map(int, data["indices"]))
            elif isinstance(data, list):
                forget_indices = list(map(int, data))
            else:
                raise ValueError(f"Unrecognised JSON structure in {spec}")
            kind = "indices"
            value = forget_indices
        else:
            try:
                arr = json.loads(spec)
                if isinstance(arr, list):
                    forget_indices = list(map(int, arr))
                    kind = "indices"
                    value = forget_indices
                else:
                    raise ValueError
            except Exception:
                raise ValueError(f"Could not parse --forget-spec={spec}")
    return kind, value, forget_indices


def load_cifar10(batch_size: int = 64) -> Tuple[datasets.CIFAR10, DataLoader, DataLoader]:
    """Load CIFAR‑10 training and validation loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(
        root=os.path.join("datasets", "cifar10"), train=True, download=True, transform=transform_train
    )
    val_dataset = datasets.CIFAR10(
        root=os.path.join("datasets", "cifar10"), train=False, download=True, transform=transform_test
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dataset, train_loader, val_loader


def compute_delta(
    before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], np.ndarray, float]:
    """Compute the parameter difference (delta) between two models."""
    delta_state: Dict[str, torch.Tensor] = {}
    delta_vectors: List[torch.Tensor] = []
    for name, param in before.items():
        if name not in after:
            continue
        d = (after[name] - param).detach().cpu()
        delta_state[name] = d
        delta_vectors.append(d.view(-1))
    if delta_vectors:
        flat = torch.cat(delta_vectors)
        delta_np = flat.numpy().astype(np.float32)
        delta_norm = float(np.linalg.norm(delta_np))
    else:
        delta_np = np.array([], dtype=np.float32)
        delta_norm = 0.0
    return delta_state, delta_np, delta_norm


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified machine unlearning baselines")
    parser.add_argument("--mu", choices=list(REGISTRY["mu"].keys()), required=True, help="Unlearning method")
    parser.add_argument("--in", dest="in_path", required=True, help="Path to the pretrained model (state dict)")
    parser.add_argument("--out", dest="out_dir", required=True, help="Output directory for results")
    parser.add_argument("--forget-spec", required=True, help="Specification of samples to forget (indices/class or JSON file)")
    parser.add_argument("--sisa-shards", type=int, default=8, help="Number of shards for SISA unlearning")
    parser.add_argument("--fisher-batches", type=int, default=64, help="Mini‑batches used to estimate the diagonal FIM")
    parser.add_argument("--fisher-scale", type=float, default=1e-3, help="Scale for Fisher regularisation term")
    parser.add_argument("--fisher-noise-std", type=float, default=0.0, help="Standard deviation of additive noise in Fisher unlearning")
    parser.add_argument("--kd-T", type=float, default=2.0, help="Temperature for KD unlearning")
    parser.add_argument("--kd-alpha", type=float, default=0.5, help="Weight on the distillation loss for KD unlearning")
    parser.add_argument("--kd-epochs", type=int, default=1, help="Fine‑tuning epochs for KD unlearning")
    parser.add_argument("--prune-ratio", type=float, default=0.2, help="Fraction of channels to prune for class‑discriminative pruning")
    parser.add_argument("--prune-score", type=str, default="tfidf", help="Type of pruning score (tfidf|grad)")
    parser.add_argument("--retrain-epochs", type=int, default=1, help="Number of epochs for exact retraining baseline")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for dataloaders")
    args = parser.parse_args(list(argv) if argv is not None else None)

    train_dataset, _train_loader_unused, val_loader = load_cifar10(batch_size=args.batch_size)
    kind, value, forget_indices = parse_forget_spec(args.forget_spec, train_dataset)
    if kind == "indices":
        keep_indices = [i for i in range(len(train_dataset)) if i not in forget_indices]
    elif kind == "class":
        keep_indices = [i for i, (_, y) in enumerate(train_dataset) if int(y) != value]
    else:
        raise RuntimeError(f"Unknown forget kind: {kind}")
    train_loader = DataLoader(Subset(train_dataset, keep_indices), batch_size=args.batch_size, shuffle=True, num_workers=2)

    state = torch.load(args.in_path, map_location="cpu")
    if isinstance(state, dict) and not any(isinstance(v, torch.Tensor) for v in state.values()):
        cand = None
        for k, v in state.items():
            if isinstance(v, dict) and all(isinstance(t, torch.Tensor) for t in v.values()):
                cand = v
                break
        if cand is None:
            raise ValueError(f"No state dict found in checkpoint {args.in_path}")
        state_dict = cand
    elif isinstance(state, dict):
        state_dict = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(state)}")

    model_before = instantiate_model(state_dict, num_classes=10)
    state_before = {k: v.clone().detach() for k, v in model_before.state_dict().items()}
    mu_cls = REGISTRY["mu"][args.mu]
    unlearner = mu_cls(model_before, train_loader, val_loader, (kind, value), args)
    start = time.time()
    model_after, metrics = unlearner.run()
    runtime = time.time() - start
    delta_state, delta_np, delta_norm = compute_delta(state_before, model_after.state_dict())

    tag = os.path.splitext(os.path.basename(args.in_path))[0]
    mu_name = args.mu
    model_out_dir = os.path.join(args.out_dir, "results", "unlearned", tag, mu_name)
    delta_out_dir = os.path.join(args.out_dir, "deltas", tag, mu_name)
    metrics_out_dir = os.path.join(args.out_dir, "metrics", "unlearn")
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(delta_out_dir, exist_ok=True)
    os.makedirs(metrics_out_dir, exist_ok=True)

    model_path = os.path.join(model_out_dir, "model_after.pt")
    torch.save({"model_state_dict": model_after.state_dict()}, model_path)
    if hasattr(unlearner, "mask"):
        torch.save(unlearner.mask, os.path.join(model_out_dir, "mask.pt"))
    torch.save(delta_state, os.path.join(delta_out_dir, "delta.pt"))
    np.save(os.path.join(delta_out_dir, "delta.npy"), delta_np)
    metrics.update(
        {
            "method": mu_name,
            "tag": tag,
            "runtime": runtime,
            "forget_kind": kind,
            "forget_value": value,
            "forget_size": len(forget_indices),
            "delta_norm": delta_norm,
        }
    )
    metrics_path = os.path.join(metrics_out_dir, f"{tag}__{mu_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Unlearning completed. Method: {mu_name}, delta norm: {delta_norm:.4f}, runtime: {runtime:.2f}s")

if __name__ == "__main__":
    main()
