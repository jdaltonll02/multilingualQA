#!/usr/bin/env bash
# ─────────────────────────────────────────
# Pipeline runner — scripts/run_all.sh
# ─────────────────────────────────────────
set -euo pipefail

# Always execute from the project root, regardless of invocation directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== [1/4] EDA ==="
python scripts/eda.py

echo "=== [2/4] Training ==="
python scripts/train.py "$@"   # pass --debug here if needed

echo "=== [3/4] Inference ==="
python scripts/predict.py

echo "=== [4/4] Evaluation ==="
python scripts/evaluate.py output/submission.csv

echo "=== Pipeline complete ==="
