#!/usr/bin/env bash

## Grid runner for machine unlearning baselines
#
# This script iterates over all supported unlearning methods and invokes
# the unified CLI with a fixed forgetting specification.  It is meant
# as a smoke test to ensure that each baseline produces the expected
# outputs (model after unlearning and parameter deltas).  Usage:
#
#     bash scripts/20_unlearn_grid.sh path/to/model_before.pt [output_dir]
#
# The second argument defaults to the current directory.  The script
# aborts on any error and will exit non‑zero if any expected file is
# missing or if the resulting update norm is zero.

set -e

MODEL_BEFORE="$1"
OUTDIR="${2:-.}"

if [[ -z "$MODEL_BEFORE" ]]; then
  echo "Usage: $0 <model_before.pt> [output_dir]"
  exit 1
fi

TAG="$(basename "${MODEL_BEFORE%.*}")"
METHODS=(retrain sisa fisher kd prune)

for MU in "${METHODS[@]}"; do
  echo "Running unlearning method: $MU"
  python -m unlearning.run \
    --mu "$MU" \
    --in "$MODEL_BEFORE" \
    --out "$OUTDIR" \
    --forget-spec class=0 \
    --sisa-shards 2 \
    --fisher-batches 4 \
    --fisher-scale 1e-3 \
    --fisher-noise-std 0.01 \
    --kd-T 2.0 \
    --kd-alpha 0.5 \
    --kd-epochs 1 \
    --prune-ratio 0.2 \
    --prune-score tfidf \
    --retrain-epochs 1 \
    --batch-size 32
  # Verify outputs
  MODEL_AFTER="${OUTDIR}/results/unlearned/${TAG}/${MU}/model_after.pt"
  DELTA_PT="${OUTDIR}/deltas/${TAG}/${MU}/delta.pt"
  DELTA_NPY="${OUTDIR}/deltas/${TAG}/${MU}/delta.npy"
  if [[ ! -f "$MODEL_AFTER" ]]; then
    echo "Error: missing ${MODEL_AFTER}"
    exit 1
  fi
  if [[ ! -f "$DELTA_PT" ]]; then
    echo "Error: missing ${DELTA_PT}"
    exit 1
  fi
  if [[ ! -f "$DELTA_NPY" ]]; then
    echo "Error: missing ${DELTA_NPY}"
    exit 1
  fi
  # Check that the delta norm is non‑zero
  python - <<'PY'
import sys, torch, numpy as np
delta = torch.load(sys.argv[1])
vec = []
for v in delta.values():
    vec.append(v.reshape(-1).numpy())
if vec:
    arr = np.concatenate(vec)
    if np.linalg.norm(arr) <= 0.0:
        raise SystemExit("Delta norm is zero")
PY
  "$DELTA_PT"
  echo "$MU completed successfully"
done
