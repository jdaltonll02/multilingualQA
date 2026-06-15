"""Training entrypoint for the multilingual QA model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import inspect
import json
import os
from collections import Counter

import pandas as pd
import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer

from src.config import load_config
from src.dataset import HealthQADataset, get_lang_label, load_tokenizer
from src.metrics import build_scorer, make_compute_metrics, per_language_eval
from src.modeling import build_training_args, load_model
from src.trainer import LanguageBalancedSeq2SeqTrainer, build_language_weights

cfg  = load_config()
DATA = cfg["data"]
TCFG = cfg["training"]
MCFG = cfg["model"]


# -- CLI -------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true",
                    help="Limit to debug_train_rows / debug_val_rows for fast iteration")
parser.add_argument("--from-tuning-results", type=str, default=None, metavar="PATH",
                    help="Path to output/tuning/results.json; applies best trial params as defaults "
                         "(CLI args still take precedence)")
parser.add_argument("--num-train-epochs",            type=float, default=None)
parser.add_argument("--learning-rate",               type=float, default=None)
parser.add_argument("--weight-decay",                type=float, default=None)
parser.add_argument("--warmup-ratio",                type=float, default=None)
parser.add_argument("--label-smoothing-factor",      type=float, default=None)
parser.add_argument("--generation-max-length",       type=int,   default=None)
parser.add_argument("--gradient-accumulation-steps", type=int,   default=None)
parser.add_argument("--seed",                        type=int,   default=None)
parser.add_argument("--output-dir",                  type=str,   default=None)
parser.add_argument("--final-model-dir",             type=str,   default=None)
parser.add_argument("--metrics-out",                 type=str,   default=None)
parser.add_argument("--save-strategy", choices=["no", "epoch", "steps"], default=None)
parser.add_argument("--skip-per-language-eval",      action="store_true")
parser.add_argument("--skip-save-model",             action="store_true")
parser.add_argument("--disable-load-best-model-at-end", action="store_true")
parser.add_argument("--balanced-sampling",           action="store_true")
parser.add_argument("--no-balanced-sampling",        action="store_true")
parser.add_argument("--balance-alpha",               type=float, default=None)
args = parser.parse_args()


# -- Load tuning results (optional) ---------------------------------------------
_tuning_defaults: dict = {}
if args.from_tuning_results:
    with open(args.from_tuning_results) as _f:
        _tdata = json.load(_f)
    _tuning_defaults = ((_tdata.get("best") or {}).get("params") or {})
    if _tuning_defaults:
        print(f"[tune] Applying best params from {args.from_tuning_results}:")
        for _k, _v in sorted(_tuning_defaults.items()):
            print(f"  {_k}: {_v}")
    else:
        print(f"[warn] No best params found in {args.from_tuning_results}; using config.yaml defaults")


# -- Resolve config (3-tier priority: CLI > tuning > config.yaml) ---------------
def _get(cli_val, cfg_val, tuning_key: str = ""):
    if cli_val is not None:
        return cli_val
    if tuning_key and tuning_key in _tuning_defaults:
        return _tuning_defaults[tuning_key]
    return cfg_val


resolved_output_dir      = _get(args.output_dir,      TCFG["output_dir"])
resolved_final_model_dir = _get(args.final_model_dir, TCFG["final_model_dir"])
resolved_save_strategy   = _get(args.save_strategy,   TCFG["save_strategy"])
resolved_load_best       = False if args.disable_load_best_model_at_end else TCFG["load_best_model_at_end"]
if resolved_save_strategy == "no":
    resolved_load_best = False
resolved_metrics_out  = _get(args.metrics_out, TCFG.get("metrics_out", "output/train_metrics.json"))
resolved_num_epochs   = float(_get(args.num_train_epochs,         TCFG["num_train_epochs"]))
resolved_lr           = float(_get(args.learning_rate,            TCFG["learning_rate"],               "learning_rate"))
resolved_wd           = float(_get(args.weight_decay,             TCFG.get("weight_decay", 0.0),       "weight_decay"))
resolved_warmup       = float(_get(args.warmup_ratio,             TCFG.get("warmup_ratio", 0.0),       "warmup_ratio"))
resolved_label_smooth = float(_get(args.label_smoothing_factor,   TCFG.get("label_smoothing_factor", 0.0), "label_smoothing_factor"))
resolved_gen_max_len  = int(_get(args.generation_max_length,      TCFG["generation_max_length"],       "generation_max_length"))
resolved_grad_accum   = int(_get(args.gradient_accumulation_steps, TCFG.get("gradient_accumulation_steps", 1), "gradient_accumulation_steps"))
resolved_seed         = int(_get(args.seed,                       TCFG.get("seed", 42)))
resolved_alpha        = float(_get(args.balance_alpha,            TCFG.get("balance_alpha", 1.0),      "balance_alpha"))

if args.balanced_sampling:
    resolved_balanced = True
elif args.no_balanced_sampling:
    resolved_balanced = False
elif "balanced_sampling" in _tuning_defaults:
    resolved_balanced = bool(_tuning_defaults["balanced_sampling"])
else:
    resolved_balanced = bool(TCFG.get("balanced_sampling", False))

os.makedirs(resolved_output_dir, exist_ok=True)
os.makedirs(resolved_final_model_dir, exist_ok=True)
if os.path.dirname(resolved_metrics_out):
    os.makedirs(os.path.dirname(resolved_metrics_out), exist_ok=True)


# -- Load data -------------------------------------------------------------------
train_df = pd.read_csv(DATA["train"])
val_df   = pd.read_csv(DATA["val"])

if args.debug:
    def _stratified_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        if len(frame) <= n:
            return frame.copy()
        tmp = frame.copy()
        tmp["_lang"] = tmp.apply(get_lang_label, axis=1)
        groups   = [g for _, g in tmp.groupby("_lang", sort=False)]
        per_lang = max(1, n // max(1, len(groups)))
        sampled  = [g.sample(n=min(per_lang, len(g)), random_state=seed) for g in groups]
        out = pd.concat(sampled)
        remaining = n - len(out)
        if remaining > 0:
            left = tmp.drop(index=out.index)
            if len(left):
                out = pd.concat([out, left.sample(n=min(remaining, len(left)), random_state=seed)])
        return out.drop(columns=["_lang"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df = _stratified_sample(train_df, TCFG["debug_train_rows"], seed=42)
    val_df   = _stratified_sample(val_df,   TCFG["debug_val_rows"],   seed=43)
    print(f"[DEBUG] {len(train_df)} train / {len(val_df)} val rows")


# -- Tokenizer, datasets, model -------------------------------------------------
tokenizer = load_tokenizer()
train_ds  = HealthQADataset(train_df, tokenizer)
val_ds    = HealthQADataset(val_df,   tokenizer)
model     = load_model(MCFG["name"])

sample_weights = None
if resolved_balanced:
    lang_counts = Counter(train_df.apply(get_lang_label, axis=1))
    print("Language distribution:", dict(sorted(lang_counts.items())))
    print(f"Balanced sampling enabled (alpha={resolved_alpha:.3f})")
    sample_weights = build_language_weights(train_df, resolved_alpha)


# -- Metrics and collator -------------------------------------------------------
scorer       = build_scorer()
pad_id       = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
vocab_max_id = int(getattr(model.config, "vocab_size", tokenizer.vocab_size or 250112)) - 1
compute_metrics = make_compute_metrics(tokenizer, scorer, pad_id, vocab_max_id)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)


# -- Training arguments ---------------------------------------------------------
training_args = build_training_args(
    output_dir=resolved_output_dir,
    num_train_epochs=resolved_num_epochs,
    per_device_train_batch_size=TCFG["per_device_train_batch_size"],
    per_device_eval_batch_size=TCFG["per_device_eval_batch_size"],
    learning_rate=resolved_lr,
    weight_decay=resolved_wd,
    warmup_ratio=resolved_warmup,
    label_smoothing_factor=resolved_label_smooth,
    optim=TCFG["optim"],
    predict_with_generate=TCFG["predict_with_generate"],
    generation_max_length=resolved_gen_max_len,
    eval_strategy=TCFG["eval_strategy"],
    save_strategy=resolved_save_strategy,
    load_best_model_at_end=resolved_load_best,
    metric_for_best_model=TCFG["metric_for_best_model"],
    fp16=TCFG["fp16"],
    bf16=TCFG.get("bf16", False),
    gradient_accumulation_steps=resolved_grad_accum,
    gradient_checkpointing=TCFG.get("gradient_checkpointing", False),
    dataloader_num_workers=TCFG.get("dataloader_num_workers", 0),
    logging_steps=TCFG["logging_steps"],
    save_total_limit=TCFG["save_total_limit"],
    report_to=TCFG["report_to"],
    seed=resolved_seed,
)


# -- Trainer --------------------------------------------------------------------
TrainerClass = LanguageBalancedSeq2SeqTrainer if resolved_balanced else Seq2SeqTrainer

trainer_kwargs = {
    "model":           model,
    "args":            training_args,
    "train_dataset":   train_ds,
    "eval_dataset":    val_ds,
    "compute_metrics": compute_metrics,
    "data_collator":   data_collator,
}
if resolved_balanced:
    trainer_kwargs["sample_weights"]       = sample_weights
    trainer_kwargs["use_weighted_sampler"] = True

sig = inspect.signature(TrainerClass.__init__).parameters
if "tokenizer" in sig:
    trainer_kwargs["tokenizer"] = tokenizer
elif "processing_class" in sig:
    trainer_kwargs["processing_class"] = tokenizer

trainer = TrainerClass(**trainer_kwargs)


# -- Train ----------------------------------------------------------------------
HAS_GPU = torch.cuda.is_available()
print(f"Training on {'GPU' if HAS_GPU else 'CPU'} | debug={args.debug}")
trainer.train()

if not args.skip_per_language_eval:
    per_language_eval(val_df, trainer, tokenizer, scorer, pad_id, vocab_max_id)


# -- Save -----------------------------------------------------------------------
if not args.skip_save_model:
    os.makedirs(resolved_final_model_dir, exist_ok=True)
    trainer.save_model(resolved_final_model_dir)
    tokenizer.save_pretrained(resolved_final_model_dir)
    print(f"\nModel saved to {resolved_final_model_dir}")
else:
    print("\nSkipping model save (--skip-save-model)")


# -- Summarize ------------------------------------------------------------------
logs    = trainer.state.log_history
best_r1 = max((e.get("eval_rouge1", 0) for e in logs), default=0)
best_rL = max((e.get("eval_rougeL", 0) for e in logs), default=0)
print(f"\nBest val ROUGE-1 : {best_r1:.4f}")
print(f"Best val ROUGE-L : {best_rL:.4f}")

metrics_payload = {
    "best_val_rouge1": float(best_r1),
    "best_val_rougeL": float(best_rL),
    "resolved_params": {
        "num_train_epochs":            resolved_num_epochs,
        "learning_rate":               resolved_lr,
        "weight_decay":                resolved_wd,
        "warmup_ratio":                resolved_warmup,
        "label_smoothing_factor":      resolved_label_smooth,
        "generation_max_length":       resolved_gen_max_len,
        "gradient_accumulation_steps": resolved_grad_accum,
        "balanced_sampling":           resolved_balanced,
        "balance_alpha":               resolved_alpha,
        "seed":                        resolved_seed,
    },
}
with open(resolved_metrics_out, "w") as f:
    json.dump(metrics_payload, f, indent=2)
print(f"Metrics written to {resolved_metrics_out}")
