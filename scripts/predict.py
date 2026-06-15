"""Inference script — generates answers for Test.csv and writes submission.csv."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import load_config
from src.dataset import COLS, MODEL_TYPE, NLLB_LANG_MAP, get_lang_label
from src.retrieval import build_retrieval_map, normalize_question

cfg  = load_config()
DATA = cfg["data"]
ICFG = cfg["inference"]
MCFG = cfg["model"]
TCFG = cfg["training"]
ECFG = cfg.get("ensemble", {})

# -- CLI ------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=str, default=None,
                    help="Override final_model_dir (e.g. point to ensemble model)")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -- Load model -----------------------------------------------------------------
model_dir = args.model_dir or TCFG["final_model_dir"]
print(f"Loading model from {model_dir} on {DEVICE} ({MODEL_TYPE}) ...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model     = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
model.eval()

# -- Load data ------------------------------------------------------------------
test_df  = pd.read_csv(DATA["test"])
train_df = pd.read_csv(DATA["train"])

q_col  = COLS["question"]
a_col  = COLS["answer"]
id_col = COLS["id"]

# -- Retrieval map (normalized) -------------------------------------------------
retrieval_map = build_retrieval_map(train_df, q_col, a_col)

# -- Build inputs and per-example language labels --------------------------------
def build_input(row) -> str:
    lang = get_lang_label(row)
    if MODEL_TYPE == "nllb":
        return str(row[q_col])        # NLLB conditions via tokenizer lang codes, not text prefix
    return f"{lang} question: {row[q_col]}"

test_df["_lang"]  = test_df.apply(get_lang_label, axis=1)
test_df["_input"] = test_df.apply(build_input, axis=1)

# -- Batched generation ---------------------------------------------------------
all_answers: list[str] = []
questions   = test_df[q_col].astype(str).str.strip().tolist()
inputs_list = test_df["_input"].tolist()
lang_list   = test_df["_lang"].tolist()
batch_size  = ICFG["batch_size"]

for i in tqdm(range(0, len(inputs_list), batch_size), desc="Generating"):
    batch_inputs = inputs_list[i : i + batch_size]
    batch_qs     = questions[i : i + batch_size]
    batch_langs  = lang_list[i : i + batch_size]

    batch_answers: list[str | None] = [None] * len(batch_inputs)
    gen_indices: list[int] = []
    gen_inputs:  list[str] = []
    gen_langs:   list[str] = []

    for j, (inp, q, lang) in enumerate(zip(batch_inputs, batch_qs, batch_langs)):
        norm_q = normalize_question(q)
        if norm_q in retrieval_map:
            batch_answers[j] = retrieval_map[norm_q]
        else:
            gen_indices.append(j)
            gen_inputs.append(inp)
            gen_langs.append(lang)

    if gen_inputs:
        if MODEL_TYPE == "nllb":
            # NLLB requires a uniform forced_bos_token_id per call, so group by language
            lang_groups: dict[str, list] = defaultdict(list)
            for k, (j, inp, lang) in enumerate(zip(gen_indices, gen_inputs, gen_langs)):
                lang_groups[lang].append((k, j, inp))

            for lang, group in lang_groups.items():
                ks, js, inps = zip(*group)
                nllb_code = NLLB_LANG_MAP.get(lang, "eng_Latn")
                tokenizer.src_lang = nllb_code

                enc = tokenizer(
                    list(inps),
                    max_length=MCFG["input_max_len"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(DEVICE)

                forced_bos_id = tokenizer.convert_tokens_to_ids(nllb_code)
                with torch.no_grad():
                    out_ids = model.generate(
                        **enc,
                        forced_bos_token_id=forced_bos_id,
                        num_beams=ICFG["num_beams"],
                        max_new_tokens=ICFG["max_new_tokens"],
                        no_repeat_ngram_size=ICFG["no_repeat_ngram_size"],
                        length_penalty=ICFG["length_penalty"],
                        early_stopping=ICFG["early_stopping"],
                    )

                for idx_in_grp, j in enumerate(js):
                    batch_answers[j] = tokenizer.decode(out_ids[idx_in_grp], skip_special_tokens=True)

        else:
            # mT5 / mBART: standard batched generation
            enc = tokenizer(
                gen_inputs,
                max_length=MCFG["input_max_len"],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(DEVICE)

            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    num_beams=ICFG["num_beams"],
                    max_new_tokens=ICFG["max_new_tokens"],
                    no_repeat_ngram_size=ICFG["no_repeat_ngram_size"],
                    length_penalty=ICFG["length_penalty"],
                    early_stopping=ICFG["early_stopping"],
                )

            for k, (j, inp) in enumerate(zip(gen_indices, gen_inputs)):
                decoded = tokenizer.decode(out_ids[k], skip_special_tokens=True)
                # Strip echoed input prefix if the model repeated the question
                prefix = inp.split("question:")[-1].strip()
                if decoded.startswith(prefix):
                    decoded = decoded[len(prefix):].strip()
                batch_answers[j] = decoded

    all_answers.extend(batch_answers)

# -- Build submission -----------------------------------------------------------
submission = pd.DataFrame({
    "ID":         test_df[id_col],
    "TargetRLF1": all_answers,
    "TargetR1F1": all_answers,
    "TargetLLM":  all_answers,
})

output_file = ICFG["output_file"]
out_dir = os.path.dirname(output_file)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
submission.to_csv(output_file, index=False)
print(f"\nSaved {output_file}  ({len(submission)} rows)")
print(submission.head())
