#!/usr/bin/env bash

# Evaluate gradient inversion attacks (DLG and GradInversion) against
# machine unlearning baselines.  For each unlearning method (retrain,
# sisa, fisher, kd, prune) this script runs four attacks: DLG with
# true gradients, DLG with Δθ pseudo-gradients, GradInversion with
# true gradients, and GradInversion with Δθ.  The resulting metrics
# are consolidated into a CSV file under results/summary.

set -e

# Arguments: <model_before.pt> [output_dir]
MODEL="$1"
OUTDIR="${2:-.}"

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <model_before.pt> [output_dir]"
  exit 1
fi

TAG="$(basename "${MODEL%.*}")"
METHODS=(retrain sisa fisher kd prune)

SUMMARY_DIR="$OUTDIR/results/summary"
mkdir -p "$SUMMARY_DIR"
CSV_PATH="$SUMMARY_DIR/attack_metrics.csv"
echo "tag,mu,method,grad_src,mse,psnr,steps,time" > "$CSV_PATH"

for MU in "${METHODS[@]}"; do
  echo "Evaluating GIA attacks for MU=$MU"
  TRUE_GRAD_FILE="$OUTDIR/results/grads/${TAG}/${MU}/grad.npy"
  DELTA_FILE="$OUTDIR/results/deltas/${TAG}/${MU}/delta.npy"
  # DLG with true gradient
  if [[ -f "$TRUE_GRAD_FILE" ]]; then
    python -m gia.dlg \
      --model "$MODEL" \
      --save-dir "$OUTDIR" \
      --steps 10 \
      --optim lbfgs \
      --grad-src true \
      --grad-file "$TRUE_GRAD_FILE"
    METRICS_PATH="$OUTDIR/results/gia/${TAG}/${MU}/dlg_true/metrics.json"
    if [[ -f "$METRICS_PATH" ]]; then
      python scripts/extract_metrics_to_csv.py "$METRICS_PATH" "$CSV_PATH"
    fi
  else
    echo "Warning: missing true gradient file $TRUE_GRAD_FILE for MU=$MU; skipping DLG true attack"
  fi
  # DLG with delta pseudo-gradient
  if [[ -f "$DELTA_FILE" ]]; then
    python -m gia.dlg \
      --model "$MODEL" \
      --save-dir "$OUTDIR" \
      --steps 10 \
      --optim lbfgs \
      --grad-src delta \
      --delta "$DELTA_FILE"
    METRICS_PATH="$OUTDIR/results/gia/${TAG}/${MU}/dlg_delta/metrics.json"
    if [[ -f "$METRICS_PATH" ]]; then
      python scripts/extract_metrics_to_csv.py "$METRICS_PATH" "$CSV_PATH"
    fi
  else
    echo "Warning: missing delta file $DELTA_FILE for MU=$MU; skipping DLG delta attack"
  fi
  # GradInversion with true gradient
  if [[ -f "$TRUE_GRAD_FILE" ]]; then
    python -m gia.gradinversion \
      --model "$MODEL" \
      --save-dir "$OUTDIR" \
      --steps 10 \
      --optim lbfgs \
      --grad-src true \
      --grad-file "$TRUE_GRAD_FILE"
    METRICS_PATH="$OUTDIR/results/gia/${TAG}/${MU}/gradinv_true/metrics.json"
    if [[ -f "$METRICS_PATH" ]]; then
      python scripts/extract_metrics_to_csv.py "$METRICS_PATH" "$CSV_PATH"
    fi
  fi
  # GradInversion with delta pseudo-gradient
  if [[ -f "$DELTA_FILE" ]]; then
    python -m gia.gradinversion \
      --model "$MODEL" \
      --save-dir "$OUTDIR" \
      --steps 10 \
      --optim lbfgs \
      --grad-src delta \
      --delta "$DELTA_FILE"
    METRICS_PATH="$OUTDIR/results/gia/${TAG}/${MU}/gradinv_delta/metrics.json"
    if [[ -f "$METRICS_PATH" ]]; then
      python scripts/extract_metrics_to_csv.py "$METRICS_PATH" "$CSV_PATH"
    fi
  fi
done

echo "GIA evaluation completed. Summary written to $CSV_PATH"
