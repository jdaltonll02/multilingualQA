#!/bin/bash
# ─────────────────────────────────────────
# Slurm pipeline orchestrator — slurm/pipeline.slurm
#
# Submits all jobs with proper dependencies so they run in order.
# Usage:
#   bash slurm/pipeline.slurm            # full run
#   bash slurm/pipeline.slurm --debug    # debug/fast-iteration mode
# ─────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

DEBUG_FLAG="${1:-}"   # pass --debug to enable debug mode for train job

mkdir -p logs output

echo "Submitting multilqa pipeline..."

# ── 1. EDA ─────────────────────────────────────────────────────────────────────
JID_EDA=$(sbatch --parsable slurm/eda.slurm)
echo "  EDA      job id: $JID_EDA"

# ── 2. Train (depends on EDA completing successfully) ──────────────────────────
JID_TRAIN=$(sbatch --parsable \
    --dependency=afterok:$JID_EDA \
    slurm/train.slurm $DEBUG_FLAG)
echo "  Train    job id: $JID_TRAIN  (after $JID_EDA)"

# ── 3. Predict (depends on Train) ─────────────────────────────────────────────
JID_PRED=$(sbatch --parsable \
    --dependency=afterok:$JID_TRAIN \
    slurm/predict.slurm)
echo "  Predict  job id: $JID_PRED  (after $JID_TRAIN)"

# ── 4. Evaluate (depends on Predict) ──────────────────────────────────────────
JID_EVAL=$(sbatch --parsable \
    --dependency=afterok:$JID_PRED \
    slurm/evaluate.slurm)
echo "  Evaluate job id: $JID_EVAL  (after $JID_PRED)"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs will appear in: logs/"
echo "Artifacts will appear in: output/"
