#!/bin/bash
#SBATCH --job-name=multilqa-predict
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --workdir=/home/jgibson2/projects/multilqa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --output=/home/jgibson2/projects/multilqa/logs/predict_%j.out
#SBATCH --error=/home/jgibson2/projects/multilqa/logs/predict_%j.err

set -euo pipefail


mkdir -p logs || true
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
VENV_PATH="/home/jgibson2/projects/multilqa/venv"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
	echo "venv activation script not found at $VENV_PATH/bin/activate"
	exit 2
fi
source "$VENV_PATH/bin/activate"

python predict.py

echo "=== Predict job finished: $(date) ==="
