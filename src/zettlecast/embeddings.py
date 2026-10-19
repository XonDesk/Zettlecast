"""
Embedding and reranking models (loaded lazily on the GPU).

The API server unloads them after MODEL_IDLE_MINUTES without use
(start_idle_unloader), so a transcription run gets the GPU memory back; the
next search loads them again (~10 s).

- Qwen3-Embedding: documents are encoded as-is, queries with the model's
  built-in "query" instruction prompt.
- Qwen3-Reranker: cross-encoder scoring of (query, passage) pairs, 0..1.
"""

import contextlib
import gc
import logging
import os
import threading
import time
from typing import Optional

import numpy as np

from .config import settings

logger = logging.getLogger(__name__)

_embedder = None
_reranker = None
_lock = threading.RLock()  # loading, using and unloading never overlap
_last_used = 0.0
_holds = 0  # GPU work in this process that must not see the cache emptied (Parakeet)


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _quiet_transformers() -> None:
    """No "Loading weights" progress bars in CLI output."""
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()


def _touch() -> None:
    global _last_used
    _last_used = time.monotonic()


def get_embedder():
    global _embedder
    _touch()
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
    _touch()
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
    with _lock:
        vectors = get_embedder().encode(
            # 8: half the activation peak of 32 (1.7 vs 3.4 GB) and no slower; during a run the
            # cached peak sits next to the LLM, and VRAM past 32 GB spills to system RAM
            texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True
        )
    return vectors.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a search query (uses the model's query instruction prompt)."""
    with _lock:
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

    with _lock:
        scores = get_reranker().predict(
            [(query, p) for p in passages],
            batch_size=16,
            activation_fn=torch.nn.Sigmoid(),
        )
    ranked = sorted(enumerate(float(s) for s in scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k] if top_k else ranked


# --- Freeing the GPU ---


def release_models(force: bool = False) -> bool:
    """
    Unload both models and return their GPU memory. False if nothing is loaded, or a run
    holds them (force: the run itself frees them before transcribing; they reload on use).
    """
    global _embedder, _reranker
    with _lock:
        if (_holds and not force) or (_embedder is None and _reranker is None):
            return False
        _embedder = _reranker = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("Search models unloaded")
    return True


@contextlib.contextmanager
def hold():
    """
    Mark GPU work in progress (a transcription run): the idle unloader waits.
    Never empty the CUDA cache between Parakeet transcribe() calls; it aborts the process.
    """
    global _holds
    with _lock:
        _holds += 1
    try:
        yield
    finally:
        with _lock:
            _holds -= 1


def start_idle_unloader(
    idle_minutes: float, check_seconds: float = 30.0, stop: Optional[threading.Event] = None
) -> Optional[threading.Thread]:
    """Background thread that unloads the models after idle_minutes without use (0 = never)."""
    if idle_minutes <= 0:
        return None
    stop = stop or threading.Event()

    def loop() -> None:
        while not stop.wait(check_seconds):
            if time.monotonic() - _last_used > idle_minutes * 60:
                release_models()

    thread = threading.Thread(target=loop, name="model-idle-unloader", daemon=True)
    thread.start()
    return thread
