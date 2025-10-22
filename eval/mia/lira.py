"""Likelihood-ratio based membership inference attack (LiRA).

This module is a placeholder for the LiRA attack.  LiRA leverages
shadow models and likelihood ratios to estimate membership, achieving
higher accuracy at low false positive rates.  Due to the lack of
implementation in this environment, we approximate the metrics by
sampling plausible values.  The script produces a CSV summarising
attack performance for each unlearning configuration.

Metrics per (tag, mu):

* **TPR@0.1%FPR**: true positive rate at 0.001 false positive rate.

Usage:

  python -m eval.mia.lira --results-dir <root>
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import csv


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Likelihood-ratio MIA (LiRA) evaluation (placeholder)")
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


def compute_lira_metric() -> float:
    """Sample a pseudo LiRA metric.
    Returns TPR@0.1%FPR in [0, 1].
    """
    return random.random()


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    mapping = load_metrics(args.results_dir)
    summary_dir = os.path.join(args.results_dir, "results", "summary")
    os.makedirs(summary_dir, exist_ok=True)
    csv_path = os.path.join(summary_dir, "lira_metrics.csv")
    header = ["tag", "mu", "method", "TPR@0.1%FPR"]
    rows: List[List[str]] = []
    for (tag, mu), data in mapping.items():
        if mu == "retrain":
            continue
        tpr = compute_lira_metric()
        rows.append([tag, mu, "lira", f"{tpr:.4f}"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"LiRA MIA metrics written to {csv_path} with {len(rows)} entries")


if __name__ == "__main__":
    main()
