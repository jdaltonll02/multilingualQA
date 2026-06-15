#!/bin/bash
#SBATCH --job-name=multilqa-eda
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --workdir=/home/jgibson2/projects/multilqa
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=cpu_qos
#SBATCH --export=ALL
#SBATCH --output=/home/jgibson2/projects/multilqa/logs/eda_%j.out
#SBATCH --error=/home/jgibson2/projects/multilqa/logs/eda_%j.err

set -euo pipefail


mkdir -p logs || true
VENV_PATH="/home/jgibson2/projects/multilqa/venv"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
	echo "venv activation script not found at $VENV_PATH/bin/activate"
	exit 2
fi
source "$VENV_PATH/bin/activate"

python eda.py

echo "=== EDA job finished: $(date) ==="
