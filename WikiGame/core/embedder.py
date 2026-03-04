# core/embedder.py
# Sentence embeddings + cosine similarity scoring

import logging
import numpy as np
from typing import Union
from functools import lru_cache

logger = logging.getLogger(__name__)

# Lazy import — sentence_transformers is heavy, only load when needed
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        import config
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


class Embedder:
    """
    Wraps sentence-transformers to embed text and compute cosine similarity.

    Usage:
        embedder = Embedder()
        target_emb = embedder.embed("Alan Turing was a mathematician and computer scientist")
        scores = embedder.score_candidates(target_emb, [
            ("Python (programming language)", "Python is a high-level programming language..."),
            ("Computer science", "Computer science is the study of computation..."),
        ])
    """

    def __init__(self):
        self.model = _get_model()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string. Returns a normalized 1D numpy array."""
        emb = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return emb

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of strings. Returns a 2D array (N, dim)."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        return embeddings

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two normalized vectors.
        Since we use normalize_embeddings=True, this is just a dot product.
        """
        return float(np.dot(a, b))

    def score_candidates(
        self,
        target_embedding: np.ndarray,
        candidates: list[tuple[str, str]],  # (title, text_to_embed)
    ) -> list[tuple[str, float]]:
        """
        Score a list of (title, text) candidates against a target embedding.

        Returns list of (title, score) sorted descending by score.
        """
        if not candidates:
            return []

        titles = [c[0] for c in candidates]
        texts = [c[1] for c in candidates]

        # Embed all candidates in one batch
        candidate_embeddings = self.embed_batch(texts)

        # Cosine similarity: target · each candidate (dot product of normalized vecs)
        scores = candidate_embeddings @ target_embedding  # shape (N,)

        scored = list(zip(titles, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def build_target_text(self, target_title: str, target_lede: str) -> str:
        """
        Build a rich text representation of the target for embedding.
        Combining title + lede gives better semantic signal than title alone.
        """
        return f"{target_title}. {target_lede}"

    def build_candidate_text(self, title: str, lede: str = "") -> str:
        """
        Build embedding text for a link candidate.
        If lede is empty (we haven't fetched that page yet), just use the title.
        """
        if lede:
            return f"{title}. {lede}"
        return title
