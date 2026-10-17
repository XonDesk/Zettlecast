"""
Embedding and reranking models (loaded lazily, kept on the GPU).

- Qwen3-Embedding: documents are encoded as-is, queries with the model's
  built-in "query" instruction prompt.
- Qwen3-Reranker: cross-encoder scoring of (query, passage) pairs, 0..1.
"""

import logging
import os
from typing import Optional

import numpy as np

from .config import settings

logger = logging.getLogger(__name__)

_embedder = None
_reranker = None


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _quiet_transformers() -> None:
    """No "Loading weights" progress bars in CLI output."""
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()


def get_embedder():
    global _embedder
    if _embedder is None:
        import torch
        from sentence_transformers import SentenceTransformer

        _quiet_transformers()
        if settings.hf_token:
            os.environ.setdefault("HF_TOKEN", settings.hf_token)
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embedder = SentenceTransformer(
            settings.embedding_model,
            device=_device(),
            model_kwargs={"torch_dtype": torch.bfloat16},
            truncate_dim=settings.embedding_dimensions,
        )
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        import torch
        from sentence_transformers import CrossEncoder

        _quiet_transformers()
        logger.info(f"Loading reranker: {settings.reranker_model}")
        _reranker = CrossEncoder(
            settings.reranker_model,
            device=_device(),
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
    return _reranker


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed passages. Returns L2-normalised float32 array (n, dim)."""
    if not texts:
        return np.zeros((0, settings.embedding_dimensions), dtype=np.float32)
    vectors = get_embedder().encode(
        texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True
    )
    return vectors.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a search query (uses the model's query instruction prompt)."""
    model = get_embedder()
    prompt_name = "query" if "query" in (model.prompts or {}) else None
    vector = model.encode(
        [text], prompt_name=prompt_name, normalize_embeddings=True, convert_to_numpy=True
    )[0]
    return vector.astype(np.float32)


def rerank(query: str, passages: list[str], top_k: Optional[int] = None) -> list[tuple[int, float]]:
    """
    Score passages against the query.

    Returns:
        (index into passages, score 0..1) pairs, best first
    """
    if not passages:
        return []
    import torch

    scores = get_reranker().predict(
        [(query, p) for p in passages],
        batch_size=16,
        activation_fn=torch.nn.Sigmoid(),
    )
    ranked = sorted(enumerate(float(s) for s in scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k] if top_k else ranked
