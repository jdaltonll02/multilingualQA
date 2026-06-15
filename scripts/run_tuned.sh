#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Full competition pipeline:
#   EDA → tune hyperparams → train with best params → ensemble → predict → eval
#
# Usage:
#   bash scripts/run_tuned.sh           # full run
#   bash scripts/run_tuned.sh --debug   # fast smoke-test on small data
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Always run from project root regardless of invocation directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== [1/6] EDA ==="
python scripts/eda.py

echo "=== [2/6] Hyperparameter tuning ==="
python scripts/tune.py "$@"

echo "=== [3/6] Training with best tuning params ==="
python scripts/train.py --from-tuning-results output/tuning/results.json "$@"

echo "=== [4/6] Checkpoint ensembling ==="
python scripts/ensemble.py

echo "=== [5/6] Inference (ensemble model) ==="
python scripts/predict.py --model-dir output/ensemble_model

echo "=== [6/6] Evaluation ==="
python scripts/evaluate.py output/submission.csv

echo "=== Full pipeline complete ==="
