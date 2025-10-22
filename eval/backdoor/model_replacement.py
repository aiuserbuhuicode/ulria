"""Backdoor evaluation via model replacement.

This module simulates the evaluation of a backdoor (model replacement)
attack in the context of machine unlearning.  The idea of model
replacement is to insert a malicious trigger into the model during
unlearning such that the model will misclassify specific trigger
inputs while retaining overall accuracy.  The metrics reported are:

* **ASR_before**: attack success rate before the trigger injection
* **ASR_after**: attack success rate after the trigger injection
* **delta_ASR**: difference ``ASR_after - ASR_before``

Since we do not have a full model or dataset in this environment, we
approximate these metrics with random values.

Usage:

  python -m eval.backdoor.model_replacement --results-dir <root>

Outputs a CSV ``results/summary/backdoor_metrics.csv`` summarising the
backdoor risk for each unlearning configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import csv


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model replacement backdoor evaluation (placeholder)")
    parser.add_argument(
        "--results-dir",
        default=".",
        help="Root directory containing results (default: current directory)",
    )
    return parser.parse_args(argv)


def load_metrics(results_dir: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    metrics_dir = os.path.join(results_dir, "results", "metrics", "unlearn")
    mapping: Dict[Tuple[str, str], Dict[str, float]] = {}
    if not os.path.isdir(metrics_dir):
        return mapping
    for fname in os.listdir(metrics_dir):
        if not fname.endswith(".json"):
            continue
        tag_mu = fname[:-5]
        if "__" not in tag_mu:
            continue
        tag, mu = tag_mu.split("__", 1)
        with open(os.path.join(metrics_dir, fname), "r") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        mapping[(tag, mu)] = data
    return mapping


def compute_backdoor_metrics() -> Tuple[float, float, float]:
    """Sample pseudo backdoor metrics.

    Returns (ASR_before, ASR_after, delta_ASR).
    """
    asr_before = random.random() * 0.1  # small baseline ASR
    improvement = random.random() * 0.5  # attack injection may drastically increase ASR
    asr_after = min(1.0, asr_before + improvement)
    delta_asr = asr_after - asr_before
    return asr_before, asr_after, delta_asr


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    mapping = load_metrics(args.results_dir)
    summary_dir = os.path.join(args.results_dir, "results", "summary")
    os.makedirs(summary_dir, exist_ok=True)
    csv_path = os.path.join(summary_dir, "backdoor_metrics.csv")
    header = ["tag", "mu", "ASR_before", "ASR_after", "delta_ASR"]
    rows: List[List[str]] = []
    for (tag, mu), data in mapping.items():
        if mu == "retrain":
            continue
        asr_before, asr_after, delta_asr = compute_backdoor_metrics()
        rows.append([
            tag,
            mu,
            f"{asr_before:.4f}",
            f"{asr_after:.4f}",
            f"{delta_asr:.4f}",
        ])
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Backdoor metrics written to {csv_path} with {len(rows)} entries")


if __name__ == "__main__":
    main()
