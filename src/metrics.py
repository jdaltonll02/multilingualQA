import numpy as np
from rouge_score import rouge_scorer

from src.dataset import HealthQADataset, get_lang_label


def build_scorer() -> rouge_scorer.RougeScorer:
    return rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)


def to_token_ids(arr, pad_id: int, vocab_max_id: int) -> np.ndarray:
    """Normalize trainer outputs to a 2D int64 token-id array safe for decoding."""
    if isinstance(arr, (tuple, list)):
        arr = arr[0]
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr.argmax(axis=-1)
    elif arr.ndim > 3:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = np.nan_to_num(arr, nan=pad_id, posinf=vocab_max_id, neginf=pad_id)
    arr = arr.astype(np.int64, copy=False)
    return np.clip(arr, 0, vocab_max_id)


def make_compute_metrics(tokenizer, scorer, pad_id: int, vocab_max_id: int):
    """Return a HuggingFace-compatible compute_metrics function."""
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds  = to_token_ids(preds, pad_id, vocab_max_id)
        labels = np.where(labels == -100, pad_id, labels)
        labels = to_token_ids(labels, pad_id, vocab_max_id)

        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
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
    return compute_metrics


def per_language_eval(val_df, trainer, tokenizer, scorer, pad_id: int, vocab_max_id: int) -> None:
    """Run per-language ROUGE breakdown on the validation set and print results."""
    print("\n── Per-language ROUGE breakdown ──")
    val_df = val_df.copy()
    val_df["_lang"] = val_df.apply(get_lang_label, axis=1)

    for lang in sorted(val_df["_lang"].unique()):
        subset = val_df[val_df["_lang"] == lang].reset_index(drop=True)
        sub_ds = HealthQADataset(subset, tokenizer)
        out    = trainer.predict(sub_ds)

        preds  = to_token_ids(out.predictions, pad_id, vocab_max_id)
        labels = np.where(out.label_ids == -100, pad_id, out.label_ids)
        labels = to_token_ids(labels, pad_id, vocab_max_id)

        dec_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        r1, rL = [], []
        for p, ref in zip(dec_preds, dec_labels):
            s = scorer.score(ref.strip(), p.strip())
            r1.append(s["rouge1"].fmeasure)
            rL.append(s["rougeL"].fmeasure)
        print(f"  {lang:10s}  ROUGE-1={np.mean(r1):.4f}  ROUGE-L={np.mean(rL):.4f}")
