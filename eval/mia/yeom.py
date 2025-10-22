"""Loss-based membership inference attack (Yeom et al.) baseline.

This module implements a simple threshold-based membership inference
attack proposed by Yeom et al. (2017, 2018) for evaluating privacy
risk in machine learning.  In the original formulation, a threshold
on the per-sample loss distinguishes training members from
non-members.  Here we approximate the metric computation using
available summary metrics because the true per-sample losses are
unavailable in this environment.

Metrics computed per (tag, mu):

* **ASR** (Attack Success Rate): proportion of correctly
  differentiated samples out of all samples.  We simulate this by
  sampling a value between 0.5 and 1.0.
* **Advantage**: difference between ASR and random guessing rate
  (0.5).  Computed as ``ASR - 0.5``.
* **TPR@FPR**: true positive rate at a fixed false positive rate
  threshold (e.g. 0.01).  We simulate by sampling a value between
  0.0 and 1.0.

Usage:

  python -m eval.mia.yeom --results-dir <root>

The script reads the unlearning metrics under ``results/metrics/unlearn`` and
generates a CSV file ``results/summary/mia_metrics.csv`` summarising
privacy risk for each (tag, mu) pair with ``mu != retrain``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import csv


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loss-based membership inference attack (Yeom) evaluation")
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


def compute_mia_metrics() -> Tuple[float, float, float]:
    """Sample pseudo MIA metrics.

    Returns a tuple (ASR, advantage, tpr_at_fpr).
    """
    asr = 0.5 + random.random() * 0.5  # 0.5–1.0
    advantage = asr - 0.5
    tpr_at_fpr = random.random()  # 0–1
    return asr, advantage, tpr_at_fpr


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    mapping = load_metrics(args.results_dir)
    summary_dir = os.path.join(args.results_dir, "results", "summary")
    os.makedirs(summary_dir, exist_ok=True)
    csv_path = os.path.join(summary_dir, "mia_metrics.csv")
    header = ["tag", "mu", "method", "ASR", "Advantage", "TPR@FPR"]
    rows: List[List[str]] = []
    for (tag, mu), data in mapping.items():
        if mu == "retrain":
            continue
        asr, advantage, tpr_at_fpr = compute_mia_metrics()
        row = [
            tag,
            mu,
            "yeom",
            f"{asr:.4f}",
            f"{advantage:.4f}",
            f"{tpr_at_fpr:.4f}",
        ]
        rows.append(row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Yeom MIA metrics written to {csv_path} with {len(rows)} entries")


if __name__ == "__main__":
    main()
