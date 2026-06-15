import re

import pandas as pd


def normalize_question(text: str) -> str:
    """Lowercase and collapse whitespace for retrieval matching."""
    text = str(text).lower().strip()
    return re.sub(r"\s+", " ", text)


def build_retrieval_map(train_df: pd.DataFrame, q_col: str, a_col: str) -> dict[str, str]:
    """Map every normalized training question to its answer."""
    result: dict[str, str] = {}
    for _, row in train_df.iterrows():
        key = normalize_question(str(row[q_col]))
        result[key] = str(row[a_col])
    return result
