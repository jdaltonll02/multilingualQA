#!/bin/bash
#SBATCH --job-name=multilqa-train
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=/home/jgibson2/projects/multilqa/logs/train_%j.out
#SBATCH --error=/home/jgibson2/projects/multilqa/logs/train_%j.err

# Pass --debug as first argument to enable fast-iteration mode:
#   sbatch slurm/train.slurm --debug

set -euo pipefail

on_exit() {
	rc=$?
	echo "=== Train job exiting: $(date) rc=${rc} job_id=${SLURM_JOB_ID:-unknown} node=${SLURMD_NODENAME:-unknown} ==="
}

on_term() {
	echo "=== Received termination signal at $(date); job_id=${SLURM_JOB_ID:-unknown} ==="
}

trap on_exit EXIT
trap on_term TERM INT USR1

mkdir -p logs output/checkpoints || true

echo "=== Train job started: $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown} SLURM_NODELIST=${SLURM_NODELIST:-unknown}"

VENV_PATH="/home/jgibson2/projects/multilqa/venv"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
	echo "venv activation script not found at $VENV_PATH/bin/activate"
	exit 2
fi

source "$VENV_PATH/bin/activate"
echo "python=$(command -v python)"
python -V

python - <<'PY'
import sys
import torch
print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
		print("CUDA is not available in this job environment. Exiting to avoid very slow CPU training.")
		sys.exit(2)
PY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py "$@"

echo "=== Train job finished: $(date) ==="
