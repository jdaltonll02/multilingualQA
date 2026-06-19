import re

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def normalize_question(text: str) -> str:
    """Lowercase and collapse whitespace for exact-match retrieval."""
    text = str(text).lower().strip()
    return re.sub(r"\s+", " ", text)


def build_retrieval_map(train_df: pd.DataFrame, q_col: str, a_col: str) -> dict[str, str]:
    """Map every normalized training question to its answer."""
    result: dict[str, str] = {}
    for _, row in train_df.iterrows():
        key = normalize_question(str(row[q_col]))
        result[key] = str(row[a_col])
    return result


class SemanticRetriever:
    """Embedding-based nearest-neighbour retriever over training Q&A pairs.

    Retrieves top_k nearest neighbours per question. Among neighbours whose
    cosine similarity >= threshold, returns the longest answer (more complete
    answers score higher on LLM judges). Falls through to generation otherwise.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        q_col: str,
        a_col: str,
        model_name: str = "sentence-transformers/LaBSE",
        threshold: float = 0.82,
        top_k: int = 3,
        encode_batch_size: int = 256,
    ):
        from sentence_transformers import SentenceTransformer

        self._threshold = threshold
        self._top_k     = max(top_k, 1)
        self._answers: list[str] = train_df[a_col].astype(str).tolist()
        questions: list[str]     = train_df[q_col].astype(str).tolist()

        print(f"[semantic retrieval] Loading encoder: {model_name}")
        self._encoder = SentenceTransformer(model_name, local_files_only=True)
        print(f"[semantic retrieval] Encoding {len(questions)} training questions...")
        embeddings: np.ndarray = self._encoder.encode(
            questions,
            batch_size=encode_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self._index = NearestNeighbors(
            n_neighbors=self._top_k, metric="cosine", algorithm="brute"
        )
        self._index.fit(embeddings)
        print(f"[semantic retrieval] Index ready (top_k={self._top_k}).")

    def retrieve_batch(
        self,
        questions: list[str],
        langs: list[str] | None = None,
        per_lang_thresholds: dict[str, float] | None = None,
    ) -> list[str | None]:
        """Return the best training answer for each question, or None if below threshold.

        Among top_k neighbours above the threshold, picks the longest answer
        (empirically correlates with completeness / LLM-judge score).
        Per-language thresholds override the global default when provided.
        """
        embs: np.ndarray = self._encoder.encode(
            questions,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # distances shape: (n_questions, top_k); sklearn cosine = 1 - sim
        distances, indices = self._index.kneighbors(embs)

        results: list[str | None] = []
        for q_idx, (dists, idxs) in enumerate(zip(distances, indices)):
            threshold = (
                per_lang_thresholds.get(langs[q_idx], self._threshold)
                if per_lang_thresholds and langs
                else self._threshold
            )
            candidates = [
                self._answers[int(idx)]
                for dist, idx in zip(dists, idxs)
                if (1.0 - float(dist)) >= threshold
            ]
            results.append(max(candidates, key=len) if candidates else None)
        return results
