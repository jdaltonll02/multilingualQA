"""Local evaluation script — ROUGE scoring against Val.csv."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from src.config import load_config
from src.dataset import get_lang_label

cfg  = load_config()
DATA = cfg["data"]
ECFG = cfg["evaluation"]
COLS = cfg["columns"]

parser = argparse.ArgumentParser()
parser.add_argument("predictions_csv", nargs="?", default=None,
                    help="Path to val predictions CSV (default: inference.val_output_file from config)")
args = parser.parse_args()

predictions_csv = args.predictions_csv or cfg["inference"].get("val_output_file", "output/val_predictions.csv")

val_df  = pd.read_csv(DATA["val"])
pred_df = pd.read_csv(predictions_csv)

id_col = COLS["id"]
a_col  = COLS["answer"]

merged = val_df.merge(pred_df[["ID", "TargetR1F1"]], on="ID", how="inner")
print(f"Evaluating {len(merged)} rows")

merged["_lang"] = merged.apply(get_lang_label, axis=1)

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)


def score_rows(refs, preds) -> tuple[float, float]:
    r1, rL = [], []
    for ref, pred in zip(refs, preds):
        s = scorer.score(str(ref).strip(), str(pred).strip())
        r1.append(s["rouge1"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return float(np.mean(r1)), float(np.mean(rL))


overall_r1, overall_rL = score_rows(merged[a_col], merged["TargetR1F1"])
llm_proxy   = (overall_r1 + overall_rL) / 2
w1, wL, wl  = ECFG["weight_rouge1"], ECFG["weight_rougeL"], ECFG["weight_llm"]
final_score = w1 * overall_r1 + wL * overall_rL + wl * llm_proxy

print(f"\nOverall  ROUGE-1 : {overall_r1:.4f}")
print(f"Overall  ROUGE-L : {overall_rL:.4f}")
print(f"LLM proxy        : {llm_proxy:.4f}")
print(f"Weighted score   : {final_score:.4f}  ({w1}*R1 + {wL}*RL + {wl}*proxy)")

print("\n── Per-language breakdown ──")
lang_results: dict = {}
for lang in sorted(merged["_lang"].unique()):
    sub = merged[merged["_lang"] == lang]
    r1, rL = score_rows(sub[a_col], sub["TargetR1F1"])
    proxy = (r1 + rL) / 2
    ws    = w1 * r1 + wL * rL + wl * proxy
    print(f"  {lang:10s}  n={len(sub):5d}  ROUGE-1={r1:.4f}  ROUGE-L={rL:.4f}  weighted={ws:.4f}")
    lang_results[lang] = {"rouge1": r1, "rougeL": rL, "llm_proxy": proxy, "weighted": ws}

results = {
    "overall": {
        "rouge1":    overall_r1,
        "rougeL":    overall_rL,
        "llm_proxy": llm_proxy,
        "weighted":  final_score,
    },
    "per_language": lang_results,
}
out_dir = os.path.dirname(ECFG["output_file"])
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
with open(ECFG["output_file"], "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {ECFG['output_file']}")
