"""
Sentence embeddings for semantic similarity.

Wraps Sentence Transformers behind a small interface so the rest of the
codebase doesn't care which embedding model is loaded, and so the model
is loaded once (lazily, then cached) rather than on every request.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)


class EmbeddingUnavailableError(Exception):
    """Raised when the embedding model can't be loaded (e.g. no internet
    on first run to download weights). Callers should fall back to the
    TF-IDF baseline in this case."""


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {CONFIG.embedding_model_name}")
    return SentenceTransformer(CONFIG.embedding_model_name)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return an (n_texts, dim) array of sentence embeddings."""
    if not texts:
        return np.zeros((0, 384))
    try:
        model = _get_model()
        embeddings = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings
    except Exception as e:  # noqa: BLE001
        raise EmbeddingUnavailableError(str(e)) from e


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, in [0, 1] after
    clamping (embeddings are normalized so raw cosine is in [-1, 1])."""
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    sim = float(np.dot(vec_a, vec_b) / denom)
    return max(0.0, min(1.0, (sim + 1) / 2 if sim < 0 else sim))


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Convenience: embed two texts and return their cosine similarity."""
    embeddings = embed_texts([text_a, text_b])
    if embeddings.shape[0] < 2:
        return 0.0
    return cosine_similarity(embeddings[0], embeddings[1])
