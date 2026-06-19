"""Inference script — generates answers for Test.csv and Val.csv."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import re
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import load_config
from src.dataset import COLS, MODEL_TYPE, NLLB_LANG_MAP, get_lang_label
from src.retrieval import SemanticRetriever, build_retrieval_map, normalize_question

cfg  = load_config()
DATA = cfg["data"]
ICFG = cfg["inference"]
MCFG = cfg["model"]
TCFG = cfg["training"]
RCFG = cfg.get("retrieval", {})

# -- CLI ------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=str, default=None,
                    help="Override final_model_dir (e.g. point to ensemble model)")
parser.add_argument("--skip-val", action="store_true",
                    help="Skip generating val predictions (val predictions are used for local eval)")
parser.add_argument("--no-semantic", action="store_true",
                    help="Disable semantic retrieval (exact-match only)")
parser.add_argument("--llm-model", type=str, default=None,
                    help="LLM model ID for TargetLLM column (and ROUGE fallback when --no-local-model). "
                         "E.g. gemini-2.5-flash or gemini-2.5-pro.")
parser.add_argument("--no-local-model", action="store_true",
                    help="Skip loading the local model; use LLM for ROUGE fallback generation too.")
parser.add_argument("--llm-provider", type=str, default=None,
                    choices=["claude", "gemini"],
                    help="LLM provider. Auto-detected from --llm-model if omitted.")
parser.add_argument("--llm-llm-col", action="store_true",
                    help="Use LLM for TargetLLM column even when retrieval succeeds.")
parser.add_argument("--llm-workers", type=int, default=2,
                    help="Concurrent LLM API threads (default 2; keep low for rate-limited APIs)")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -- LLM API setup --------------------------------------------------------------
def _detect_provider(model_id: str) -> str:
    if model_id.startswith("claude"):
        return "claude"
    if model_id.startswith("gemini"):
        return "gemini"
    raise ValueError(f"Cannot auto-detect provider for model '{model_id}'. "
                     "Pass --llm-provider claude|gemini explicitly.")

llm_client = None
llm_provider: str | None = None

if args.llm_model:
    llm_provider = args.llm_provider or _detect_provider(args.llm_model)
    try:
        if llm_provider == "claude":
            import anthropic
            llm_client = anthropic.Anthropic()
        elif llm_provider == "gemini":
            import vertexai
            from vertexai.generative_models import GenerativeModel
            vertexai.init(
                project=os.environ["GOOGLE_CLOUD_PROJECT"],
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
            llm_client = GenerativeModel(args.llm_model)
        print(f"LLM generation enabled: provider={llm_provider}  model={args.llm_model}  "
              f"llm_col_always={args.llm_llm_col}")
    except Exception as e:
        print(f"WARNING: Failed to initialise LLM client ({e}). Falling back to primary model.")
        args.llm_model = None

_LANG_DISPLAY = {
    "akan": "Akan (Twi)",
    "amharic": "Amharic",
    "luganda": "Luganda",
    "swahili": "Swahili",
    "english": "English",
}

_FEWSHOT_CACHE: dict[str, list[tuple[str, str]]] = {}

def _get_fewshot(train_df_local, subset: str, n: int = 3) -> list[tuple[str, str]]:
    if subset not in _FEWSHOT_CACHE:
        rows = train_df_local[train_df_local["subset"] == subset]
        sample = rows.sample(min(n * 4, len(rows)), random_state=42).head(n)
        _FEWSHOT_CACHE[subset] = list(zip(sample[COLS["question"]], sample[COLS["answer"]]))
    return _FEWSHOT_CACHE[subset]


def _call_llm(question: str, lang: str, subset: str, context: str | None = None) -> str:
    """Call the configured LLM with exponential backoff on rate-limit errors."""
    import time
    lang_display = _LANG_DISPLAY.get(lang, lang.capitalize())
    fewshots = _get_fewshot(train_df, subset)
    shots_text = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in fewshots)
    context_block = (f"\nHere is a related answer from a knowledge base that may help:\n{context}\n"
                     if context else "")
    system = (f"You are a health information assistant. "
              f"Answer the following question in {lang_display}. "
              f"Be concise and specific. Match the style of the examples.")
    user = (f"Here are example Q&A pairs in {lang_display}:\n\n{shots_text}\n\n"
            f"{context_block}"
            f"Now answer this question in {lang_display}:\n{question}")

    max_retries = 6
    wait = 10  # seconds; doubles each retry
    for attempt in range(max_retries):
        try:
            if llm_provider == "claude":
                resp = llm_client.messages.create(
                    model=args.llm_model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": user}],
                    system=system,
                )
                return resp.content[0].text.strip()
            elif llm_provider == "gemini":
                resp = llm_client.generate_content(f"{system}\n\n{user}")
                return resp.text.strip()
        except Exception as e:
            if "429" in str(e) or "Resource exhausted" in str(e) or "rate" in str(e).lower():
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    wait *= 2
                    continue
            raise
    return ""


def _call_llm_batch(items: list[tuple[int, str, str, str, str | None]],
                    workers: int) -> dict[int, str]:
    """items = list of (idx, question, lang, subset, context_or_None).
    Returns {idx: answer}."""
    results: dict[int, str] = {}

    def _worker(item):
        idx, q, lang, subset, ctx = item
        try:
            return idx, _call_llm(q, lang, subset, ctx)
        except Exception as e:
            print(f"  LLM error on idx={idx}: {e}")
            return idx, ""

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, it): it[0] for it in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"LLM API ({llm_provider})"):
            idx, ans = fut.result()
            results[idx] = ans
    return results


# -- Load primary model (skipped when LLM handles all generation) ---------------
model_dir = str(Path(args.model_dir or TCFG["final_model_dir"]).resolve())
tokenizer = None
model     = None
amh_model     = None
amh_tokenizer = None

if not args.no_local_model:
    print(f"Loading primary model from {model_dir} on {DEVICE} ({MODEL_TYPE}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True).to(DEVICE)
    model.eval()
    model.generation_config.max_length = ICFG["max_new_tokens"] + MCFG["input_max_len"]

    _amh_model_name = MCFG.get("amharic_model")
    if _amh_model_name:
        print(f"Loading Amharic router model: {_amh_model_name} ...")
        amh_tokenizer = AutoTokenizer.from_pretrained(_amh_model_name, src_lang="amh_Ethi", local_files_only=True)
        amh_model     = AutoModelForSeq2SeqLM.from_pretrained(_amh_model_name, local_files_only=True).to(DEVICE)
        amh_model.eval()
        amh_model.generation_config.max_length = ICFG["max_new_tokens"] + MCFG["input_max_len"]
        print("Amharic router model ready.")
else:
    print(f"Skipping primary model load — {llm_provider} ({args.llm_model}) handles ROUGE fallback generation.")

# -- Load data ------------------------------------------------------------------
train_df = pd.read_csv(DATA["train"])
q_col    = COLS["question"]
a_col    = COLS["answer"]
id_col   = COLS["id"]

retrieval_map = build_retrieval_map(train_df, q_col, a_col)
print(f"Exact-match retrieval map: {len(retrieval_map)} entries")

semantic_retriever: SemanticRetriever | None = None
per_lang_thresholds: dict[str, float] | None = None

if not args.no_semantic and RCFG:
    _thresh_path = Path(RCFG.get("thresholds_file", "output/retrieval_thresholds.json"))
    if _thresh_path.exists():
        per_lang_thresholds = json.loads(_thresh_path.read_text()).get("per_language")
        print(f"Loaded per-language thresholds from {_thresh_path}: {per_lang_thresholds}")
    else:
        print(f"No tuned thresholds at {_thresh_path}; using global threshold={RCFG.get('similarity_threshold', 0.82)}")

    semantic_retriever = SemanticRetriever(
        train_df=train_df,
        q_col=q_col,
        a_col=a_col,
        model_name=RCFG.get("semantic_model", "sentence-transformers/LaBSE"),
        threshold=float(RCFG.get("similarity_threshold", 0.82)),
        top_k=int(RCFG.get("top_k", 3)),
        encode_batch_size=int(RCFG.get("encode_batch_size", 256)),
    )

_EXTRA_ID_RE = re.compile(r"<extra_id_\d+>")


def clean(text: str) -> str:
    return _EXTRA_ID_RE.sub("", text).strip()


def build_input(row) -> str:
    lang = get_lang_label(row)
    if MODEL_TYPE == "nllb":
        return str(row[q_col])
    return f"{lang} question: {row[q_col]}"


def _generate_nllb(model_, tokenizer_, questions: list[str], lang: str) -> list[str]:
    """Run NLLB-200 generation for a single language group."""
    nllb_code = NLLB_LANG_MAP.get(lang, "eng_Latn")
    tokenizer_.src_lang = nllb_code
    enc = tokenizer_(
        questions,
        max_length=MCFG["input_max_len"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    forced_bos_id = tokenizer_.convert_tokens_to_ids(nllb_code)
    with torch.no_grad():
        out_ids = model_.generate(
            **enc,
            forced_bos_token_id=forced_bos_id,
            num_beams=ICFG["num_beams"],
            max_new_tokens=ICFG["max_new_tokens"],
            no_repeat_ngram_size=ICFG["no_repeat_ngram_size"],
            length_penalty=ICFG["length_penalty"],
            early_stopping=ICFG["early_stopping"],
        )
    return [clean(tokenizer_.decode(ids, skip_special_tokens=True)) for ids in out_ids]


def generate_answers(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (answers_rouge, answers_llm).

    answers_rouge  — used for TargetRLF1 and TargetR1F1 (retrieval preferred for ROUGE)
    answers_llm    — used for TargetLLM (LLM API when --llm-model is set)
    """
    df = df.copy()
    df["_lang"]   = df.apply(get_lang_label, axis=1)
    df["_input"]  = df.apply(build_input, axis=1)
    df["_subset"] = df["subset"] if "subset" in df.columns else df["ID"].str.extract(r"ID_(?:TS|VL)_(\w+_\w+)_")[0]

    n = len(df)
    answers_rouge: list[str | None] = [None] * n
    answers_llm:   list[str | None] = [None] * n

    questions   = df[q_col].astype(str).str.strip().tolist()
    inputs_list = df["_input"].tolist()
    lang_list   = df["_lang"].tolist()
    subset_list = df["_subset"].tolist()
    batch_size  = ICFG["batch_size"]

    retrieved_exact = 0
    retrieved_sem   = 0
    generated       = 0

    # Track indices still needing generation after retrieval (for all rows)
    global_offset = 0

    for i in tqdm(range(0, n, batch_size), desc="Retrieval + generation"):
        batch_inputs  = inputs_list[i : i + batch_size]
        batch_qs      = questions[i : i + batch_size]
        batch_langs   = lang_list[i : i + batch_size]
        batch_subsets = subset_list[i : i + batch_size]
        bsz = len(batch_inputs)

        batch_rouge: list[str | None] = [None] * bsz
        # Track per-batch: (local_j, q, lang, subset, retrieved_answer_or_None)
        retrieval_results: list[tuple[int, str, str, str, str | None]] = []

        # ── Step 1: exact-match retrieval ─────────────────────────────────────
        remaining: list[tuple[int, str, str, str]] = []
        for j, (inp, q, lang, sub) in enumerate(
                zip(batch_inputs, batch_qs, batch_langs, batch_subsets)):
            norm_q = normalize_question(q)
            if norm_q in retrieval_map:
                batch_rouge[j] = retrieval_map[norm_q]
                retrieval_results.append((j, q, lang, sub, retrieval_map[norm_q]))
                retrieved_exact += 1
            else:
                remaining.append((j, inp, q, lang))

        # ── Step 2: semantic retrieval ────────────────────────────────────────
        if semantic_retriever is not None and remaining:
            sem_answers = semantic_retriever.retrieve_batch(
                [r[2] for r in remaining],
                langs=[r[3] for r in remaining],
                per_lang_thresholds=per_lang_thresholds,
            )
            still_remaining: list[tuple[int, str, str, str]] = []
            for (j, inp, q, lang), sem_ans in zip(remaining, sem_answers):
                sub = batch_subsets[j]
                if sem_ans is not None:
                    batch_rouge[j] = sem_ans
                    retrieval_results.append((j, q, lang, sub, sem_ans))
                    retrieved_sem += 1
                else:
                    retrieval_results.append((j, q, lang, sub, None))
                    still_remaining.append((j, inp, q, lang))
            remaining = still_remaining
        else:
            for j, inp, q, lang in remaining:
                sub = batch_subsets[j]
                retrieval_results.append((j, q, lang, sub, None))

        # ── Step 3: generation for unmatched items ────────────────────────────
        if remaining:
            generated += len(remaining)

            if llm_client is not None and args.no_local_model:
                # LLM replaces model generation for the fallback (only when --no-local-model)
                llm_items = [
                    (j, q, lang, batch_subsets[j], None)
                    for j, inp, q, lang in remaining
                ]
                llm_gen_results = _call_llm_batch(llm_items, args.llm_workers)
                for j, ans in llm_gen_results.items():
                    batch_rouge[j] = ans
                    # Update retrieval_results so LLM column also gets this answer
                    for idx_r, (rj, rq, rl, rs, _) in enumerate(retrieval_results):
                        if rj == j:
                            retrieval_results[idx_r] = (rj, rq, rl, rs, ans)
                            break
            else:
                # Original model generation path
                amh_items   = [(j, q, lang) for j, inp, q, lang in remaining
                               if lang == "amharic" and amh_model is not None]
                other_items = [(j, inp, q, lang) for j, inp, q, lang in remaining
                               if not (lang == "amharic" and amh_model is not None)]

                if amh_items:
                    amh_js = [j for j, q, lang in amh_items]
                    amh_qs = [q for j, q, lang in amh_items]
                    decoded = _generate_nllb(amh_model, amh_tokenizer, amh_qs, "amharic")
                    for j, text in zip(amh_js, decoded):
                        batch_rouge[j] = text

                if other_items:
                    gen_indices = [r[0] for r in other_items]
                    gen_inputs  = [r[1] for r in other_items]
                    gen_langs   = [r[3] for r in other_items]

                    if MODEL_TYPE == "nllb":
                        lang_groups: dict[str, list] = defaultdict(list)
                        for k, (j, inp, lang) in enumerate(zip(gen_indices, gen_inputs, gen_langs)):
                            lang_groups[lang].append((k, j, inp))
                        for lang, group in lang_groups.items():
                            ks, js, inps = zip(*group)
                            decoded = _generate_nllb(model, tokenizer, list(inps), lang)
                            for idx_in_grp, j in enumerate(js):
                                batch_rouge[j] = decoded[idx_in_grp]
                    else:
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
                            decoded = clean(tokenizer.decode(out_ids[k], skip_special_tokens=True))
                            prefix = inp.split("question:")[-1].strip()
                            if decoded.startswith(prefix):
                                decoded = decoded[len(prefix):].strip()
                            batch_rouge[j] = decoded

        # ── Step 4: TargetLLM column via LLM API ─────────────────────────────
        if llm_client is not None and args.llm_llm_col:
            # Always call LLM for LLM column, passing retrieval as context
            llm_items = [
                (j, q, lang, sub, ret_ans)
                for j, q, lang, sub, ret_ans in retrieval_results
            ]
            llm_results = _call_llm_batch(llm_items, args.llm_workers)
            batch_llm: list[str | None] = [None] * bsz
            for j, ans in llm_results.items():
                batch_llm[j] = ans
        else:
            # LLM column = same as ROUGE column (no extra Claude call)
            batch_llm = list(batch_rouge)

        answers_rouge[i : i + bsz] = batch_rouge
        answers_llm[i : i + bsz]   = batch_llm

    total = retrieved_exact + retrieved_sem + generated
    print(
        f"Retrieval summary: exact={retrieved_exact} ({retrieved_exact/total:.1%}), "
        f"semantic={retrieved_sem} ({retrieved_sem/total:.1%}), "
        f"generated={generated} ({generated/total:.1%})"
    )
    return answers_rouge, answers_llm


def save_submission(df: pd.DataFrame, answers_rouge: list[str], answers_llm: list[str],
                    output_file: str) -> None:
    def _clean(answers):
        return [str(a).replace("\n", " ").replace("\r", " ").strip() for a in answers]

    out = pd.DataFrame({
        "ID":         df[id_col],
        "TargetRLF1": _clean(answers_rouge),
        "TargetR1F1": _clean(answers_rouge),
        "TargetLLM":  _clean(answers_llm),
    })
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_file, index=False)
    print(f"Saved {output_file}  ({len(out)} rows)")
    print(out.head())


# -- Test predictions -----------------------------------------------------------
print("\n=== Test predictions ===")
test_df = pd.read_csv(DATA["test"])
test_answers_rouge, test_answers_llm = generate_answers(test_df)
save_submission(test_df, test_answers_rouge, test_answers_llm, ICFG["output_file"])

# -- Val predictions ------------------------------------------------------------
if not args.skip_val:
    print("\n=== Val predictions (for local evaluation) ===")
    val_df = pd.read_csv(DATA["val"])
    val_answers_rouge, val_answers_llm = generate_answers(val_df)
    val_out = ICFG.get("val_output_file", "output/val_predictions.csv")
    save_submission(val_df, val_answers_rouge, val_answers_llm, val_out)
