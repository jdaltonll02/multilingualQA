# multilqa

A multilingual health question-answering training pipeline built around Hugging Face Transformers and PyTorch.

## What this repo does

- trains `google/mt5-base` on multilingual health QA pairs
- uses language-aware prompts and balanced sampling across languages
- supports stable BF16 training for mT5
- includes a random search tuning workflow
- provides SLURM launchers for cluster training and tuning

## Key files

- `config.yaml` - central configuration for data, model, training, tuning, and inference
- `dataset.py` - dataset wrapper and language resolution logic
- `train.py` - main training entrypoint with CLI overrides
- `tune.py` - random search tuner for hyperparameter experiments
- `slurm/train.slurm` - SLURM job launcher for training
- `slurm/tune.slurm` - SLURM job launcher for tuning
- `docs/TUTORIAL.md` - usage guide and architectural overview

## Getting started

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick commands

```bash
python train.py
python train.py --debug
python tune.py
python tune.py --debug
sbatch slurm/train.slurm
sbatch slurm/tune.slurm
```

## Notes

- Use `--debug` for fast iteration on a small, language-stratified subset.
- SLURM scripts expect a local virtual environment at `venv/`.
- `config.yaml` contains the current search space and default defaults.

## Documentation

See `docs/TUTORIAL.md` for a step-by-step guide, architecture summary, and workflow details.
