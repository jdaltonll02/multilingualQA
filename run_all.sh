#!/usr/bin/env bash
# ─────────────────────────────────────────
# Pipeline runner — run_all.sh
# ─────────────────────────────────────────
set -euo pipefail

echo "=== [1/4] EDA ==="
python eda.py

echo "=== [2/4] Training ==="
python train.py "$@"          # pass --debug here if needed

echo "=== [3/4] Inference ==="
python predict.py

echo "=== [4/4] Evaluation ==="
python evaluate.py output/submission.csv

echo "=== Pipeline complete ==="
