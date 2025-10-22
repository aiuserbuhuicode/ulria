#!/usr/bin/env bash

# Evaluate method-centric metrics, membership inference attacks, and backdoor attacks.
#
# This script orchestrates the evaluation pipeline for machine unlearning
# experiments.  It runs the method-centric metrics computation, the
# membership inference attacks (Yeom and optionally LiRA), and the
# model replacement backdoor evaluation.  All generated metrics are
# stored under ``results/summary``.
#
# Usage:
#   bash scripts/40_eval_method_centric.sh [results_dir]
#
# If ``results_dir`` is omitted, the current directory is used.

set -e

OUTDIR="${1:-.}"
echo "Running method-centric metrics evaluation..."
python -m eval.metrics_method --results-dir "$OUTDIR"
echo "Running Yeom membership inference attack evaluation..."
python -m eval.mia.yeom --results-dir "$OUTDIR"
echo "Running LiRA membership inference attack evaluation..."
python -m eval.mia.lira --results-dir "$OUTDIR" || echo "LiRA attack skipped"
echo "Running backdoor model replacement evaluation..."
python -m eval.backdoor.model_replacement --results-dir "$OUTDIR"
echo "Method-centric evaluation completed.  Results are stored in $OUTDIR/results/summary."
