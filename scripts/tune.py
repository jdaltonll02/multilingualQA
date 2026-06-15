"""Random-search hyperparameter tuning for multilingual QA training."""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SCRIPTS_DIR = Path(__file__).resolve().parent

import yaml


def is_mt5_model(model_name: str) -> bool:
    n = (model_name or "").lower()
    return "mt5" in n or "/mt5" in n


def sample_choice(rng: random.Random, values):
    return values[rng.randrange(len(values))]


def to_flag_list(params):
    flags = [
        "--learning-rate", str(params["learning_rate"]),
        "--weight-decay", str(params["weight_decay"]),
        "--warmup-ratio", str(params["warmup_ratio"]),
        "--label-smoothing-factor", str(params["label_smoothing_factor"]),
        "--generation-max-length", str(params["generation_max_length"]),
        "--gradient-accumulation-steps", str(params["gradient_accumulation_steps"]),
        "--balance-alpha", str(params["balance_alpha"]),
    ]
    if params["balanced_sampling"]:
        flags.append("--balanced-sampling")
    else:
        flags.append("--no-balanced-sampling")
    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--trial-epochs", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="Use debug-sized data for fast tuning")
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.get("model", {}).get("name", "")
    mt5_mode = is_mt5_model(model_name)

    tuning_cfg = cfg.get("tuning", {})
    trials = args.trials if args.trials is not None else int(tuning_cfg.get("trials", 8))
    trial_epochs = args.trial_epochs if args.trial_epochs is not None else float(tuning_cfg.get("trial_epochs", 2))
    seed = args.seed if args.seed is not None else int(tuning_cfg.get("trial_seed", 42))

    rng = random.Random(seed)
    out_dir = Path("output/tuning")
    out_dir.mkdir(parents=True, exist_ok=True)

    label_smoothing_values = tuning_cfg.get("label_smoothing_factors", [0.0])
    if mt5_mode:
        label_smoothing_values = [0.0]

    search_space = {
        "learning_rate": tuning_cfg.get("learning_rates", [5.0e-4]),
        "weight_decay": tuning_cfg.get("weight_decays", [0.0]),
        "warmup_ratio": tuning_cfg.get("warmup_ratios", [0.03]),
        "label_smoothing_factor": label_smoothing_values,
        "generation_max_length": tuning_cfg.get("generation_max_lengths", [64]),
        "gradient_accumulation_steps": tuning_cfg.get("gradient_accumulation_steps", [2]),
        "balanced_sampling": tuning_cfg.get("balanced_sampling_options", [False]),
        "balance_alpha": tuning_cfg.get("balance_alphas", [1.0]),
    }

    if mt5_mode:
        print(
            "[tune] mT5 detected: forcing label_smoothing_factor=0.0 "
            "to avoid decoder_input_ids training crash"
        )

    trial_results = []

    for idx in range(1, trials + 1):
        trial_name = f"trial_{idx:02d}"
        trial_dir = out_dir / trial_name
        metrics_path = out_dir / f"{trial_name}_metrics.json"

        params = {
            "learning_rate": sample_choice(rng, search_space["learning_rate"]),
            "weight_decay": sample_choice(rng, search_space["weight_decay"]),
            "warmup_ratio": sample_choice(rng, search_space["warmup_ratio"]),
            "label_smoothing_factor": sample_choice(rng, search_space["label_smoothing_factor"]),
            "generation_max_length": sample_choice(rng, search_space["generation_max_length"]),
            "gradient_accumulation_steps": sample_choice(rng, search_space["gradient_accumulation_steps"]),
            "balanced_sampling": sample_choice(rng, search_space["balanced_sampling"]),
            "balance_alpha": sample_choice(rng, search_space["balance_alpha"]),
        }

        cmd = [
            sys.executable,
            str(_SCRIPTS_DIR / "train.py"),
            "--num-train-epochs", str(trial_epochs),
            "--output-dir", str(trial_dir / "checkpoints"),
            "--final-model-dir", str(trial_dir / "final_model"),
            "--save-strategy", "no",
            "--disable-load-best-model-at-end",
            "--skip-save-model",
            "--skip-per-language-eval",
            "--metrics-out", str(metrics_path),
            "--seed", str(seed + idx),
        ]
        if args.debug:
            cmd.append("--debug")
        cmd.extend(to_flag_list(params))

        print(f"\n=== {trial_name} / {trials} ===", flush=True)
        print("params:", json.dumps(params, sort_keys=True), flush=True)
        proc = subprocess.run(cmd)

        trial_record = {
            "trial": trial_name,
            "params": params,
            "exit_code": proc.returncode,
            "best_val_rouge1": None,
            "best_val_rougeL": None,
            "score": -1.0,
        }

        if proc.returncode == 0 and metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            trial_record["best_val_rouge1"] = float(m.get("best_val_rouge1", 0.0))
            trial_record["best_val_rougeL"] = float(m.get("best_val_rougeL", 0.0))
            trial_record["score"] = float(m.get("best_val_rougeL", 0.0))
        elif proc.returncode != 0:
            print(
                f"[tune] {trial_name} failed with exit code {proc.returncode} "
                f"(params={json.dumps(params, sort_keys=True)})",
                flush=True,
            )

        trial_results.append(trial_record)

    trial_results.sort(key=lambda x: x["score"], reverse=True)
    best = trial_results[0] if trial_results else None

    summary = {
        "trials": trial_results,
        "best": best,
        "optimized_metric": "best_val_rougeL",
    }

    summary_path = out_dir / "results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Tuning complete ===")
    print(f"Results saved to {summary_path}")
    if best is not None:
        print("Best trial:", json.dumps(best, indent=2, sort_keys=True))
        print("\nSuggested full-train overrides:")
        print("python scripts/train.py ", " ".join(to_flag_list(best["params"])))


if __name__ == "__main__":
    main()
