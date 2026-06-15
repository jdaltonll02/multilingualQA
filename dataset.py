# ─────────────────────────────────────────
# Dataset & Preprocessing — dataset.py
# ─────────────────────────────────────────
import yaml
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from langdetect import detect

# ── Load config ────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_NAME     = cfg["model"]["name"]
INPUT_MAX_LEN  = cfg["model"]["input_max_len"]
TARGET_MAX_LEN = cfg["model"]["target_max_len"]
COLS           = cfg["columns"]
LANG_MAP       = cfg["language_map"]    # subset prefix → label  e.g. "Aka" → "akan"
LDET_MAP       = cfg["langdetect_map"]  # langdetect code → label


# ── Language helpers ───────────────────────────────────────────────────────────
def subset_to_lang(subset_val: str) -> str:
    """Derive language label from subset string like 'Aka_Gha'."""
    prefix = str(subset_val).split("_")[0]
    return LANG_MAP.get(prefix, "english")


def detect_lang(text: str) -> str:
    """Fallback: detect language from raw text."""
    try:
        code = detect(str(text))
        return LDET_MAP.get(code, "english")
    except Exception:
        return "english"


def get_lang_label(row) -> str:
    """Resolve language label from a DataFrame row."""
    subset_col = COLS["subset"]
    if subset_col in row.index and str(row[subset_col]) not in ("nan", "None", ""):
        return subset_to_lang(row[subset_col])
    return detect_lang(row[COLS["question"]])


# ── Dataset class ──────────────────────────────────────────────────────────────
class HealthQADataset(Dataset):
    """
    Wraps a pandas DataFrame of health QA pairs.
    Set `is_test=True` for inference (no labels).
    """

    def __init__(self, dataframe, tokenizer, is_test: bool = False):
        self.tokenizer = tokenizer
        self.is_test   = is_test
        self.pad_id    = tokenizer.pad_token_id

        self.inputs  = []
        self.targets = []

        q_col = COLS["question"]
        a_col = COLS["answer"]

        for _, row in dataframe.iterrows():
            lang = get_lang_label(row)
            text = f"{lang} question: {row[q_col]}"
            self.inputs.append(text)
            if not is_test:
                self.targets.append(str(row[a_col]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.inputs[idx],
            max_length=INPUT_MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if not self.is_test:
            label_enc = self.tokenizer(
                text_target=self.targets[idx],
                max_length=TARGET_MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = label_enc["input_ids"].squeeze(0)
            labels[labels == self.pad_id] = -100
            item["labels"] = labels

        return item


# ── Convenience loader ─────────────────────────────────────────────────────────
def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)
