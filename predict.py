# ─────────────────────────────────────────
# Inference Script — predict.py
# ─────────────────────────────────────────
import os
import yaml
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

from dataset import get_lang_label, COLS

# ── Load config ────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA  = cfg["data"]
ICFG  = cfg["inference"]
MCFG  = cfg["model"]
TCFG  = cfg["training"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model & tokenizer ─────────────────────────────────────────────────────
model_dir = TCFG["final_model_dir"]
print(f"Loading model from {model_dir} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model     = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
model.eval()

# ── Load data ──────────────────────────────────────────────────────────────────
test_df  = pd.read_csv(DATA["test"])
train_df = pd.read_csv(DATA["train"])

q_col = COLS["question"]
a_col = COLS["answer"]
id_col = COLS["id"]

# ── Build retrieval lookup (free ROUGE points for exact-match questions) ────────
retrieval_map = dict(zip(
    train_df[q_col].astype(str).str.strip(),
    train_df[a_col].astype(str),
))

# ── Build prefixed inputs ──────────────────────────────────────────────────────
def build_input(row) -> str:
    lang = get_lang_label(row)
    return f"{lang} question: {row[q_col]}"

test_df["_input"] = test_df.apply(build_input, axis=1)

# ── Generation in batches ──────────────────────────────────────────────────────
all_answers = []
questions   = test_df[q_col].astype(str).str.strip().tolist()
inputs_list = test_df["_input"].tolist()
batch_size  = ICFG["batch_size"]

for i in tqdm(range(0, len(inputs_list), batch_size), desc="Generating"):
    batch_inputs = inputs_list[i : i + batch_size]
    batch_qs     = questions[i : i + batch_size]

    batch_answers = [None] * len(batch_inputs)
    gen_indices: list[int] = []
    gen_inputs:  list[str] = []

    for j, (inp, q) in enumerate(zip(batch_inputs, batch_qs)):
        # ── Retrieval fallback ────────────────────────────────────────────────
        if q in retrieval_map:
            batch_answers[j] = retrieval_map[q]
        else:
            gen_indices.append(j)
            gen_inputs.append(inp)

    # ── Batch model generation for all non-retrieved items ────────────────────
    if gen_inputs:
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
            # Strip echoed input prefix if model repeated it
            prefix = inp.split("question:")[-1].strip()
            if decoded.startswith(prefix):
                decoded = decoded[len(prefix):].strip()
            batch_answers[j] = decoded

    all_answers.extend(batch_answers)

# ── Build submission ───────────────────────────────────────────────────────────
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
print("\nFirst 5 rows:")
print(submission.head())
