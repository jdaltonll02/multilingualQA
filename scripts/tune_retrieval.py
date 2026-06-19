"""
Tune per-language semantic retrieval thresholds on the validation set.

For each language and threshold candidate we estimate the combined ROUGE-L:

    combined = (n_retrieved * rouge_retrieved + n_remaining * rouge_model) / n_total

Among top_k neighbours above the threshold the longest answer is chosen,
matching the same selection rule used at inference time.

Writes: output/retrieval_thresholds.json
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from src.config import load_config
from src.dataset import COLS, get_lang_label
from src.retrieval import SemanticRetriever

cfg  = load_config()
DATA = cfg["data"]
RCFG = cfg.get("retrieval", {})
TOP_K = int(RCFG.get("top_k", 3))

# Model-only ROUGE-L baseline per language (mT0-large fine-tuned, no retrieval).
# These approximate the generation-only score; update after retraining.
MODEL_ROUGE_L_BASELINE: dict[str, float] = {
    "akan":     0.255,
    "amharic":  0.035,
    "english":  0.296,
    "luganda":  0.171,
    "swahili":  0.322,
}

THRESHOLDS = [round(t, 2) for t in np.arange(0.70, 0.99, 0.02)]
q_col = COLS["question"]
a_col = COLS["answer"]

scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def rouge_l(pred: str, ref: str) -> float:
    return scorer_obj.score(ref, pred)["rougeL"].fmeasure


# ── Build semantic index (top_k neighbours) ───────────────────────────────────
train_df = pd.read_csv(DATA["train"])
val_df   = pd.read_csv(DATA["val"])

retriever = SemanticRetriever(
    train_df=train_df,
    q_col=q_col,
    a_col=a_col,
    model_name=RCFG.get("semantic_model", "sentence-transformers/LaBSE"),
    threshold=0.0,   # retrieve everything; thresholds applied manually below
    top_k=TOP_K,
    encode_batch_size=int(RCFG.get("encode_batch_size", 256)),
)

# ── Encode all val questions and get top_k neighbours ─────────────────────────
print("Computing similarities for all val questions...")
val_questions = val_df[q_col].astype(str).tolist()
val_langs     = val_df.apply(get_lang_label, axis=1).tolist()
val_refs      = val_df[a_col].astype(str).tolist()

embs = retriever._encoder.encode(
    val_questions,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
)
# distances shape: (n_val, top_k)
distances, indices = retriever._index.kneighbors(embs)

# Build per-question list of (similarity, answer) for all top_k neighbours
all_candidates: list[list[tuple[float, str]]] = []
for dists, idxs in zip(distances, indices):
    all_candidates.append([
        (1.0 - float(d), retriever._answers[int(i)])
        for d, i in zip(dists, idxs)
    ])

# ── Sweep thresholds per language ─────────────────────────────────────────────
print("\nSweeping thresholds...\n")

results: dict[str, dict] = {}
lang_set = sorted(set(val_langs))

for lang in lang_set:
    mask    = [i for i, l in enumerate(val_langs) if l == lang]
    n_total = len(mask)
    baseline = MODEL_ROUGE_L_BASELINE.get(lang, 0.20)

    best_thresh = None
    best_score  = -1.0
    rows = []

    for t in THRESHOLDS:
        retrieved_rouges = []
        n_retrieved = 0

        for i in mask:
            # Among top_k neighbours, pick those above threshold
            valid = [ans for sim, ans in all_candidates[i] if sim >= t]
            if valid:
                best_ans = max(valid, key=len)
                retrieved_rouges.append(rouge_l(best_ans, val_refs[i]))
                n_retrieved += 1

        if n_retrieved == 0:
            ret_rouge = 0.0
            combined  = baseline
        else:
            ret_rouge = float(np.mean(retrieved_rouges))
            combined  = (n_retrieved * ret_rouge + (n_total - n_retrieved) * baseline) / n_total

        rows.append({
            "threshold":      t,
            "n_retrieved":    n_retrieved,
            "pct_retrieved":  n_retrieved / n_total,
            "ret_rougeL":     ret_rouge,
            "combined_rougeL": combined,
        })

        if combined > best_score:
            best_score  = combined
            best_thresh = t

    results[lang] = {
        "best_threshold":      best_thresh,
        "best_combined_rougeL": best_score,
        "sweep": rows,
    }
    print(f"{lang:10s}  baseline={baseline:.3f}  best_threshold={best_thresh:.2f}"
          f"  combined={best_score:.3f}")

# ── Write output ──────────────────────────────────────────────────────────────
out_path = Path(RCFG.get("thresholds_file", "output/retrieval_thresholds.json"))
out_path.parent.mkdir(parents=True, exist_ok=True)

thresholds_out = {lang: v["best_threshold"] for lang, v in results.items()}
full_out = {"per_language": thresholds_out, "detail": results}

with open(out_path, "w") as f:
    json.dump(full_out, f, indent=2)

print(f"\nPer-language thresholds: {thresholds_out}")
print(f"Written to {out_path}")
