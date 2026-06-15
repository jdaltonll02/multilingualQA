#!/bin/bash
#SBATCH --job-name=multilqa-tune
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=/home/jgibson2/projects/multilqa/logs/tune_%j.out
#SBATCH --error=/home/jgibson2/projects/multilqa/logs/tune_%j.err

set -euo pipefail

mkdir -p logs output/tuning || true

echo "=== Tuning job started: $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

VENV_PATH="/home/jgibson2/projects/multilqa/venv"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "venv activation script not found at $VENV_PATH/bin/activate"
  exit 2
fi

source "$VENV_PATH/bin/activate"
echo "python=$(command -v python)"
python -V

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pass --debug as first argument to do fast trialing on debug subset.
python tune.py "$@"

echo "=== Tuning job finished: $(date) ==="
