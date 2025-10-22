"""Gradient inversion via generative image prior (GradInversion) baseline.

This module provides a lightweight command-line interface for the
GradInversion attack (NeurIPS 2021).  GradInversion reconstructs
input images from aggregated gradients using a generative prior and
total variation regularisation.  In this simplified version we do not
include the heavy optimisation due to the absence of PyTorch in this
environment.  Instead we generate a synthetic reconstruction and
export simple metrics, enabling downstream pipelines to compare
between true gradients and Δθ negative controls.

References:
    - Yin, H., Gao, Y, Liu, Y., et al. “See through Gradients: Image
      Batch Recovery via GradInversion”. NeurIPS, 2021.  Official
      code: https://github.com/ml-postech/gradient-inversion-generative-image-prior
    - UIA paper suggests using Δθ as a degenerate gradient source to
      demonstrate that gradient inversion is not equivalent to
      unlearning.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict

import numpy as np
from PIL import Image


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GradInversion gradient inversion attack")
    parser.add_argument("--model", required=True, help="Path to the model checkpoint (unused in dummy implementation)")
    parser.add_argument("--save-dir", required=True, help="Directory to save reconstructed image and metrics")
    parser.add_argument("--steps", type=int, default=10, help="Number of optimisation steps (dummy)")
    parser.add_argument("--optim", choices=["lbfgs", "adam"], default="lbfgs", help="Optimiser for gradient matching (dummy)")
    parser.add_argument("--grad-src", choices=["true", "delta"], required=True, help="Source of gradients: true or delta pseudo-gradients")
    parser.add_argument("--grad-file", help="Path to true gradient numpy file (required if --grad-src=true)")
    parser.add_argument("--delta", help="Path to Δθ numpy file (required if --grad-src=delta)")
    parser.add_argument("--tv", action="store_true", help="Enable total variation prior (not implemented)")
    parser.add_argument("--bn-stat", action="store_true", help="Use batch norm statistics prior (not implemented)")
    args = parser.parse_args(argv)
    if args.grad_src == "true" and not args.grad_file:
        parser.error("--grad-file is required when --grad-src=true")
    if args.grad_src == "delta" and not args.delta:
        parser.error("--delta is required when --grad-src=delta")
    return args


def load_gradient(path: str) -> np.ndarray:
    return np.load(path)


def reconstruct_image(args: argparse.Namespace) -> Dict[str, Any]:
    start_time = time.time()
    if args.grad_src == "true":
        _ = load_gradient(args.grad_file)
        source_tag = "true"
        tag, mu = infer_tag_mu_from_path(args.grad_file)
    else:
        _ = load_gradient(args.delta)
        source_tag = "delta"
        tag, mu = infer_tag_mu_from_path(args.delta)
    if tag is None:
        tag = os.path.splitext(os.path.basename(args.model))[0]
    if mu is None:
        mu = "unknown"
    try:
        import torch  # type: ignore
        _torch_available = True
    except Exception:
        _torch_available = False
    out_dir = os.path.join(args.save_dir, "results", "gia", tag, mu, f"gradinv_{source_tag}")
    os.makedirs(out_dir, exist_ok=True)
    if _torch_available:
        recon = (torch.rand(3, 32, 32) * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    else:
        recon = (np.random.rand(32, 32, 3) * 255.0).astype(np.uint8)
    img_norm = recon.astype(np.float32) / 255.0
    mse = float(np.mean(img_norm ** 2))
    psnr = float("inf") if mse == 0.0 else 10.0 * np.log10(1.0 / mse)
    elapsed = time.time() - start_time
    img = Image.fromarray(recon)
    img_path = os.path.join(out_dir, "reconstructed.png")
    img.save(img_path)
    metrics = {
        "psnr": psnr,
        "mse": mse,
        "steps": args.steps,
        "time": elapsed,
        "grad_src": source_tag,
        "method": "gradinv",
        "tag": tag,
        "mu": mu,
    }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved reconstructed image to {img_path} and metrics to {metrics_path}")
    return metrics


def infer_tag_mu_from_path(path: str) -> tuple[None | str, None | str]:
    parts = path.replace("\\", "/").split("/")
    if "deltas" in parts:
        i = parts.index("deltas")
        if i + 2 < len(parts):
            return parts[i + 1], parts[i + 2]
    if "grads" in parts:
        i = parts.index("grads")
        if i + 2 < len(parts):
            return parts[i + 1], parts[i + 2]
    return None, None


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    reconstruct_image(args)


if __name__ == "__main__":
    main()
