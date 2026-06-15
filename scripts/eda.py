"""Exploratory data analysis — language distribution, answer lengths, overlap."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from langdetect import detect

from src.config import load_config

cfg  = load_config()
DATA = cfg["data"]
COLS = cfg["columns"]
LANG = cfg["language_map"]
LDET = cfg["langdetect_map"]
EDA  = cfg["eda"]

for p in (EDA["report_file"], EDA["plot_file"]):
    out_dir = os.path.dirname(p)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

train = pd.read_csv(DATA["train"])
val   = pd.read_csv(DATA["val"])
test  = pd.read_csv(DATA["test"])

splits = {"Train": train, "Val": val, "Test": test}

report_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    report_lines.append(str(msg))


def subset_to_lang(subset_val: str) -> str:
    prefix = str(subset_val).split("_")[0]
    return LANG.get(prefix, "unknown")


def infer_lang(text: str) -> str:
    try:
        code = detect(str(text))
        return LDET.get(code, code)
    except Exception:
        return "unknown"


# -- Basic stats ----------------------------------------------------------------
for name, df in splits.items():
    log(f"\n{'='*40}")
    log(f"{name}  shape: {df.shape}")
    log(df.dtypes.to_string())
    log("\nNull counts:")
    log(df.isnull().sum().to_string())

# -- Language distribution ------------------------------------------------------
log("\n" + "="*40)
log("Language distribution")

for name, df in splits.items():
    subset_col = COLS["subset"]
    if subset_col in df.columns:
        log(f"\n{name} — language from '{subset_col}' column:")
        log(df[subset_col].apply(subset_to_lang).value_counts().to_string())
    else:
        log(f"\n{name} — no '{subset_col}' column; inferring from {EDA['lang_sample_size']} samples")
        q_col  = COLS["question"]
        sample = df[q_col].dropna().sample(min(EDA["lang_sample_size"], len(df)), random_state=42)
        log(sample.apply(infer_lang).value_counts().to_string())

# -- Answer length distribution -------------------------------------------------
ans_col    = COLS["answer"]
subset_col = COLS["subset"]

if ans_col in train.columns:
    train["_lang"]    = train[subset_col].apply(subset_to_lang)
    train["_ans_len"] = train[ans_col].astype(str).apply(len)

    fig, ax = plt.subplots(figsize=(10, 5))
    for lang in sorted(train["_lang"].unique()):
        subset = train.loc[train["_lang"] == lang, "_ans_len"]
        ax.hist(subset, bins=50, alpha=0.5, label=lang)
    ax.set_xlabel("Answer length (chars)")
    ax.set_ylabel("Count")
    ax.set_title("Answer length distribution per language")
    ax.legend()
    plt.tight_layout()
    plt.savefig(EDA["plot_file"])
    log(f"\nSaved {EDA['plot_file']}")

# -- Duplicate questions (Train ∩ Test) -----------------------------------------
q_col    = COLS["question"]
train_qs = set(train[q_col].dropna().str.strip())
test_qs  = set(test[q_col].dropna().str.strip())
overlap  = train_qs & test_qs
log(f"\nExact duplicate questions (Train ∩ Test): {len(overlap)}")
log("(These are free ROUGE points via retrieval fallback in predict.py)")

# -- Save report ----------------------------------------------------------------
with open(EDA["report_file"], "w") as f:
    f.write("\n".join(report_lines))
log(f"\nSaved {EDA['report_file']}")
