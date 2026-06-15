"""Checkpoint ensembling — average weights of the top-k checkpoints by eval metric."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import load_config

cfg  = load_config()
TCFG = cfg["training"]
ECFG = cfg.get("ensemble", {})


def checkpoint_metric(ckpt_dir: Path, metric: str) -> float:
    """Read the last logged value of `metric` from a checkpoint's trainer_state.json."""
    state_file = ckpt_dir / "trainer_state.json"
    if not state_file.exists():
        return float("-inf")
    with open(state_file) as f:
        state = json.load(f)
    for entry in reversed(state.get("log_history", [])):
        if metric in entry:
            return float(entry[metric])
    return float("-inf")


def find_and_rank_checkpoints(output_dir: str, metric: str) -> list[Path]:
    """Return checkpoint dirs sorted best-first by `metric`."""
    ckpt_dirs = [
        p for p in Path(output_dir).iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-")
    ]
    if not ckpt_dirs:
        return []
    ranked = sorted(ckpt_dirs, key=lambda d: checkpoint_metric(d, metric), reverse=True)
    return ranked


def average_state_dicts(checkpoint_dirs: list[Path]) -> dict:
    """Load and arithmetically average the state dicts of multiple checkpoints."""
    print(f"Averaging {len(checkpoint_dirs)} checkpoints:")
    for d in checkpoint_dirs:
        score = checkpoint_metric(d, ECFG.get("metric", "eval_rougeL"))
        print(f"  {d.name}  ({ECFG.get('metric', 'eval_rougeL')}={score:.4f})")

    averaged: dict | None = None
    for ckpt_dir in checkpoint_dirs:
        model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_dir))
        state = {k: v.float().clone() for k, v in model.state_dict().items()}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if averaged is None:
            averaged = state
        else:
            for k in averaged:
                averaged[k] += state[k]

    n = len(checkpoint_dirs)
    for k in averaged:
        averaged[k] /= n
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=ECFG.get("top_k", 3),
                        help="Number of best checkpoints to average")
    parser.add_argument("--metric", type=str, default=ECFG.get("metric", "eval_rougeL"),
                        help="Trainer metric used to rank checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default=TCFG["output_dir"],
                        help="Directory containing checkpoint-N subdirectories")
    parser.add_argument("--output-dir", type=str, default=ECFG.get("output_dir", "output/ensemble_model"),
                        help="Where to save the ensembled model")
    args = parser.parse_args()

    ranked = find_and_rank_checkpoints(args.checkpoint_dir, args.metric)
    if not ranked:
        print(f"No checkpoints found in {args.checkpoint_dir}. Run train.py first.")
        return

    selected = ranked[: args.top_k]
    if len(selected) < 2:
        print(f"Only {len(selected)} checkpoint(s) found — need at least 2 to ensemble. "
              f"Increase save_total_limit in config.yaml or run more training epochs.")
        return

    averaged_state = average_state_dicts(selected)

    # Build output model from the best checkpoint (preserves architecture config)
    print(f"\nBuilding ensembled model ...")
    best_ckpt = selected[0]
    model = AutoModelForSeq2SeqLM.from_pretrained(str(best_ckpt))
    model.load_state_dict({k: v.to(next(model.parameters()).dtype) for k, v in averaged_state.items()})

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

    # Copy tokenizer from the best checkpoint
    tokenizer = AutoTokenizer.from_pretrained(str(best_ckpt))
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nEnsemble model saved to {args.output_dir}")
    print(f"Run inference with: python scripts/predict.py --model-dir {args.output_dir}")


if __name__ == "__main__":
    main()
