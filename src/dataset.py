import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from langdetect import detect

from src.config import load_config

cfg = load_config()

MODEL_NAME     = cfg["model"]["name"]
INPUT_MAX_LEN  = cfg["model"]["input_max_len"]
TARGET_MAX_LEN = cfg["model"]["target_max_len"]
COLS           = cfg["columns"]
LANG_MAP       = cfg["language_map"]
LDET_MAP       = cfg["langdetect_map"]


def subset_to_lang(subset_val: str) -> str:
    prefix = str(subset_val).split("_")[0]
    return LANG_MAP.get(prefix, "english")


def detect_lang(text: str) -> str:
    try:
        code = detect(str(text))
        return LDET_MAP.get(code, "english")
    except Exception:
        return "english"


def get_lang_label(row) -> str:
    subset_col = COLS["subset"]
    if subset_col in row.index and str(row[subset_col]) not in ("nan", "None", ""):
        return subset_to_lang(row[subset_col])
    return detect_lang(row[COLS["question"]])


class HealthQADataset(Dataset):
    """Wraps a pandas DataFrame of health QA pairs. Set is_test=True for inference."""

    def __init__(self, dataframe, tokenizer, is_test: bool = False):
        self.tokenizer = tokenizer
        self.is_test   = is_test
        self.pad_id    = tokenizer.pad_token_id

        q_col = COLS["question"]
        a_col = COLS["answer"]

        self.inputs:  list[str] = []
        self.targets: list[str] = []

        for _, row in dataframe.iterrows():
            lang = get_lang_label(row)
            self.inputs.append(f"{lang} question: {row[q_col]}")
            if not is_test:
                self.targets.append(str(row[a_col]))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.inputs[idx],
            max_length=INPUT_MAX_LEN,
            padding=False,
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
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            labels = label_enc["input_ids"].squeeze(0)
            labels[labels == self.pad_id] = -100
            item["labels"] = labels
        return item


def load_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(MODEL_NAME)
