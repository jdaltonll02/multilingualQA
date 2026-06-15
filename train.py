"""Training entrypoint for the multilingual QA model.

This script supports:
- Regular training/evaluation
- Optional language-balanced sampling
- CLI hyperparameter overrides for tuning workflows
"""

import argparse
import inspect
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import yaml
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

from dataset import HealthQADataset, get_lang_label, load_tokenizer

# -- Load config -----------------------------------------------------------------
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA = cfg["data"]
TCFG = cfg["training"]
MCFG = cfg["model"]


# -- CLI -------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug",
    action="store_true",
    help="Limit to debug_train_rows / debug_val_rows for fast iteration",
)
parser.add_argument("--num-train-epochs", type=float, default=None)
parser.add_argument("--learning-rate", type=float, default=None)
parser.add_argument("--weight-decay", type=float, default=None)
parser.add_argument("--warmup-ratio", type=float, default=None)
parser.add_argument("--label-smoothing-factor", type=float, default=None)
parser.add_argument("--generation-max-length", type=int, default=None)
parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--final-model-dir", type=str, default=None)
parser.add_argument("--metrics-out", type=str, default=None)
parser.add_argument("--save-strategy", type=str, choices=["no", "epoch", "steps"], default=None)
parser.add_argument("--skip-per-language-eval", action="store_true")
parser.add_argument("--skip-save-model", action="store_true")
parser.add_argument("--disable-load-best-model-at-end", action="store_true")
parser.add_argument("--balanced-sampling", action="store_true")
parser.add_argument("--no-balanced-sampling", action="store_true")
parser.add_argument("--balance-alpha", type=float, default=None)
args = parser.parse_args()


# -- Helpers ---------------------------------------------------------------------
def override_or_cfg(cli_value, cfg_value):
    return cfg_value if cli_value is None else cli_value


def build_language_weights(frame: pd.DataFrame, alpha: float) -> np.ndarray:
    """Create inverse-frequency sample weights for language-balanced sampling."""
    langs = frame.apply(get_lang_label, axis=1)
    counts = Counter(langs)
    inv = {lang: 1.0 / (count ** alpha) for lang, count in counts.items()}
    weights = np.array([inv[lang] for lang in langs], dtype=np.float64)
    # Normalize around 1.0 to keep scale stable.
    return weights / weights.mean()


# -- Resolve config with CLI overrides ------------------------------------------
resolved_output_dir = override_or_cfg(args.output_dir, TCFG["output_dir"])
resolved_final_model_dir = override_or_cfg(args.final_model_dir, TCFG["final_model_dir"])
resolved_save_strategy = override_or_cfg(args.save_strategy, TCFG["save_strategy"])
resolved_load_best_model_at_end = False if args.disable_load_best_model_at_end else TCFG["load_best_model_at_end"]
if resolved_save_strategy == "no":
    resolved_load_best_model_at_end = False

resolved_metrics_out = override_or_cfg(args.metrics_out, TCFG.get("metrics_out", "output/train_metrics.json"))
resolved_num_train_epochs = float(override_or_cfg(args.num_train_epochs, TCFG["num_train_epochs"]))
resolved_learning_rate = float(override_or_cfg(args.learning_rate, TCFG["learning_rate"]))
resolved_weight_decay = float(override_or_cfg(args.weight_decay, TCFG.get("weight_decay", 0.0)))
resolved_warmup_ratio = float(override_or_cfg(args.warmup_ratio, TCFG.get("warmup_ratio", 0.0)))
resolved_label_smoothing = float(
    override_or_cfg(args.label_smoothing_factor, TCFG.get("label_smoothing_factor", 0.0))
)
resolved_generation_max_length = int(override_or_cfg(args.generation_max_length, TCFG["generation_max_length"]))
resolved_grad_accum = int(
    override_or_cfg(args.gradient_accumulation_steps, TCFG.get("gradient_accumulation_steps", 1))
)
resolved_seed = int(override_or_cfg(args.seed, TCFG.get("seed", 42)))

balanced_sampling_cfg = TCFG.get("balanced_sampling", False)
if args.balanced_sampling:
    resolved_balanced_sampling = True
elif args.no_balanced_sampling:
    resolved_balanced_sampling = False
else:
    resolved_balanced_sampling = bool(balanced_sampling_cfg)
resolved_balance_alpha = float(override_or_cfg(args.balance_alpha, TCFG.get("balance_alpha", 1.0)))

# Ensure artifact directories exist
os.makedirs(resolved_output_dir, exist_ok=True)
os.makedirs(resolved_final_model_dir, exist_ok=True)
if os.path.dirname(resolved_metrics_out):
    os.makedirs(os.path.dirname(resolved_metrics_out), exist_ok=True)


# -- Load data -------------------------------------------------------------------
train_df = pd.read_csv(DATA["train"])
val_df = pd.read_csv(DATA["val"])

if args.debug:
    def stratified_debug_sample(frame: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
        if len(frame) <= n_rows:
            return frame.copy()

        tmp = frame.copy()
        tmp["_lang"] = tmp.apply(get_lang_label, axis=1)
        groups = [g for _, g in tmp.groupby("_lang", sort=False)]
        n_langs = max(1, len(groups))
        per_lang = max(1, n_rows // n_langs)

        sampled = []
        for grp in groups:
            take = min(per_lang, len(grp))
            sampled.append(grp.sample(n=take, random_state=seed))

        out = pd.concat(sampled, axis=0)
        remaining = n_rows - len(out)
        if remaining > 0:
            left = tmp.drop(index=out.index)
            if len(left) > 0:
                out = pd.concat([out, left.sample(n=min(remaining, len(left)), random_state=seed)], axis=0)

        out = out.drop(columns=["_lang"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return out

    train_df = stratified_debug_sample(train_df, TCFG["debug_train_rows"], seed=42)
    val_df = stratified_debug_sample(val_df, TCFG["debug_val_rows"], seed=43)
    print(f"[DEBUG] Using {len(train_df)} train / {len(val_df)} val rows")


# -- Tokenizer & datasets --------------------------------------------------------
tokenizer = load_tokenizer()
train_ds = HealthQADataset(train_df, tokenizer)
val_ds = HealthQADataset(val_df, tokenizer)
train_sample_weights = None

if resolved_balanced_sampling:
    train_lang_counts = Counter(train_df.apply(get_lang_label, axis=1))
    print("Language distribution in training set:")
    for lang, cnt in sorted(train_lang_counts.items()):
        print(f"  {lang:10s} -> {cnt}")
    print(f"Using language-balanced sampling with alpha={resolved_balance_alpha:.3f}")
    train_sample_weights = build_language_weights(train_df, resolved_balance_alpha)


# -- Model -----------------------------------------------------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(MCFG["name"])
# mT5 checkpoints for this task can carry distinct lm_head/shared weights.
model.config.tie_word_embeddings = False


# -- Metrics ---------------------------------------------------------------------
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
VOCAB_MAX_ID = int(getattr(model.config, "vocab_size", tokenizer.vocab_size or 250112)) - 1


def to_token_ids(arr):
    """Normalize trainer outputs to a 2D int64 token-id array safe for decoding."""
    if isinstance(arr, (tuple, list)):
        arr = arr[0]

    arr = np.asarray(arr)

    # If logits are returned (batch, seq, vocab), convert to token IDs first.
    if arr.ndim == 3:
        arr = arr.argmax(axis=-1)
    elif arr.ndim > 3:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.ndim == 1:
        arr = arr[None, :]

    # Make sure values are finite integers and inside tokenizer vocab bounds.
    arr = np.nan_to_num(arr, nan=PAD_ID, posinf=VOCAB_MAX_ID, neginf=PAD_ID)
    arr = arr.astype(np.int64, copy=False)
    arr = np.clip(arr, 0, VOCAB_MAX_ID)
    return arr


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = to_token_ids(preds)
    labels = np.where(labels == -100, PAD_ID, labels)
    labels = to_token_ids(labels)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    r1_scores, rL_scores = [], []
    for pred, ref in zip(decoded_preds, decoded_labels):
        s = scorer.score(ref.strip(), pred.strip())
        r1_scores.append(s["rouge1"].fmeasure)
        rL_scores.append(s["rougeL"].fmeasure)

    return {
        "rouge1": float(np.mean(r1_scores)),
        "rougeL": float(np.mean(rL_scores)),
    }


# -- Training arguments -----------------------------------------------------------
HAS_GPU = torch.cuda.is_available()

training_args = Seq2SeqTrainingArguments(
    output_dir=resolved_output_dir,
    num_train_epochs=resolved_num_train_epochs,
    per_device_train_batch_size=TCFG["per_device_train_batch_size"],
    per_device_eval_batch_size=TCFG["per_device_eval_batch_size"],
    learning_rate=resolved_learning_rate,
    weight_decay=resolved_weight_decay,
    warmup_ratio=resolved_warmup_ratio,
    label_smoothing_factor=resolved_label_smoothing,
    optim=TCFG["optim"],
    predict_with_generate=TCFG["predict_with_generate"],
    generation_max_length=resolved_generation_max_length,
    eval_strategy=TCFG["eval_strategy"],
    save_strategy=resolved_save_strategy,
    load_best_model_at_end=resolved_load_best_model_at_end,
    metric_for_best_model=TCFG["metric_for_best_model"],
    fp16=TCFG["fp16"] and HAS_GPU,
    bf16=TCFG.get("bf16", False) and HAS_GPU,
    gradient_accumulation_steps=resolved_grad_accum,
    gradient_checkpointing=TCFG.get("gradient_checkpointing", False),
    dataloader_num_workers=TCFG.get("dataloader_num_workers", 0),
    dataloader_pin_memory=HAS_GPU,
    logging_steps=TCFG["logging_steps"],
    save_total_limit=TCFG["save_total_limit"],
    report_to=TCFG["report_to"],
    seed=resolved_seed,
    data_seed=resolved_seed,
)


class LanguageBalancedSeq2SeqTrainer(Seq2SeqTrainer):
    """Trainer that uses WeightedRandomSampler for language balancing."""

    def __init__(self, *args, sample_weights=None, use_weighted_sampler=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights
        self.use_weighted_sampler = use_weighted_sampler

    def get_train_dataloader(self):
        if not self.use_weighted_sampler or self.sample_weights is None:
            return super().get_train_dataloader()

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(self.sample_weights, dtype=torch.double),
            num_samples=len(self.sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_num_workers > 0,
        )


TrainerClass = LanguageBalancedSeq2SeqTrainer if resolved_balanced_sampling else Seq2SeqTrainer

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_ds,
    "eval_dataset": val_ds,
    "compute_metrics": compute_metrics,
}
if resolved_balanced_sampling:
    trainer_kwargs["sample_weights"] = train_sample_weights
    trainer_kwargs["use_weighted_sampler"] = True

# Transformers API changed from `tokenizer` to `processing_class` in newer versions.
trainer_sig = inspect.signature(TrainerClass.__init__).parameters
if "tokenizer" in trainer_sig:
    trainer_kwargs["tokenizer"] = tokenizer
elif "processing_class" in trainer_sig:
    trainer_kwargs["processing_class"] = tokenizer

trainer = TrainerClass(**trainer_kwargs)


# -- Per-language breakdown helper ----------------------------------------------
def per_language_eval():
    print("\n── Per-language ROUGE breakdown ──")
    val_df["_lang"] = val_df.apply(get_lang_label, axis=1)

    for lang in sorted(val_df["_lang"].unique()):
        subset = val_df[val_df["_lang"] == lang].reset_index(drop=True)
        sub_ds = HealthQADataset(subset, tokenizer)
        out = trainer.predict(sub_ds)
        preds = to_token_ids(out.predictions)
        labels = np.where(out.label_ids == -100, PAD_ID, out.label_ids)
        labels = to_token_ids(labels)

        dec_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        r1, rL = [], []
        for p, r in zip(dec_preds, dec_labels):
            s = scorer.score(r.strip(), p.strip())
            r1.append(s["rouge1"].fmeasure)
            rL.append(s["rougeL"].fmeasure)
        print(f"  {lang:10s}  ROUGE-1={np.mean(r1):.4f}  ROUGE-L={np.mean(rL):.4f}")


# -- Train -----------------------------------------------------------------------
print(f"Training on {'GPU' if HAS_GPU else 'CPU'} | debug={args.debug}")
trainer.train()

if not args.skip_per_language_eval:
    per_language_eval()


# -- Save ------------------------------------------------------------------------
if not args.skip_save_model:
    os.makedirs(resolved_final_model_dir, exist_ok=True)
    trainer.save_model(resolved_final_model_dir)
    tokenizer.save_pretrained(resolved_final_model_dir)
    print(f"\nModel saved to {resolved_final_model_dir}")
else:
    print("\nSkipping model save (--skip-save-model)")


# -- Summarize -------------------------------------------------------------------
logs = trainer.state.log_history
best_r1 = max((e.get("eval_rouge1", 0) for e in logs), default=0)
best_rL = max((e.get("eval_rougeL", 0) for e in logs), default=0)
print(f"\nBest val ROUGE-1 : {best_r1:.4f}")
print(f"Best val ROUGE-L : {best_rL:.4f}")

metrics_payload = {
    "best_val_rouge1": float(best_r1),
    "best_val_rougeL": float(best_rL),
    "resolved_params": {
        "num_train_epochs": resolved_num_train_epochs,
        "learning_rate": resolved_learning_rate,
        "weight_decay": resolved_weight_decay,
        "warmup_ratio": resolved_warmup_ratio,
        "label_smoothing_factor": resolved_label_smoothing,
        "generation_max_length": resolved_generation_max_length,
        "gradient_accumulation_steps": resolved_grad_accum,
        "balanced_sampling": resolved_balanced_sampling,
        "balance_alpha": resolved_balance_alpha,
        "seed": resolved_seed,
    },
}
with open(resolved_metrics_out, "w") as f:
    json.dump(metrics_payload, f, indent=2)
print(f"Metrics written to {resolved_metrics_out}")
