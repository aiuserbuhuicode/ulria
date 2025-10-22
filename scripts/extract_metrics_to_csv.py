#!/usr/bin/env python3

"""Utility script to append metrics from a JSON file to a CSV.

This script reads a JSON file containing keys ``tag``, ``mu``,
``method``, ``grad_src``, ``mse``, ``psnr``, ``steps``, and ``time``
and appends a line to a CSV file provided as the second argument.
"""

import json
import sys


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: extract_metrics_to_csv.py <metrics.json> <csv_path>")
        sys.exit(1)
    metrics_path = sys.argv[1]
    csv_path = sys.argv[2]
    with open(metrics_path, "r") as f:
        m = json.load(f)
    with open(csv_path, "a") as f:
        f.write(f"{m['tag']},{m['mu']},{m['method']},{m['grad_src']},{m['mse']},{m['psnr']},{m['steps']},{m['time']}\n")


if __name__ == "__main__":
    main()
