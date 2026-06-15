import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Compatibility shim — source of truth is src/dataset.py
from src.dataset import (  # noqa: F401
    COLS,
    LANG_MAP,
    LDET_MAP,
    MODEL_NAME,
    INPUT_MAX_LEN,
    TARGET_MAX_LEN,
    HealthQADataset,
    detect_lang,
    get_lang_label,
    load_tokenizer,
    subset_to_lang,
)
