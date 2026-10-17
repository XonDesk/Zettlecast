"""The API frees the GPU when search is idle, but never during a transcription run."""

import threading
import time

from zettlecast import embeddings


def test_idle_models_are_released_but_not_during_a_run(monkeypatch):
    monkeypatch.setattr(embeddings, "_embedder", object())
    monkeypatch.setattr(embeddings, "_reranker", object())
    with embeddings.hold():
        assert embeddings.release_models() is False  # a transcription run holds the GPU state
    assert embeddings._embedder is not None
    assert embeddings.release_models() is True
    assert embeddings._embedder is None and embeddings._reranker is None
    assert embeddings.release_models() is False  # nothing left to free


def test_unloader_waits_for_idle_time(monkeypatch):
    released = []
    monkeypatch.setattr(embeddings, "release_models", lambda: released.append(1))
    monkeypatch.setattr(embeddings, "_last_used", time.monotonic())
    stop = threading.Event()
    thread = embeddings.start_idle_unloader(idle_minutes=0.5 / 60, check_seconds=0.05, stop=stop)
    try:
        time.sleep(0.2)
        assert not released  # used 0.2 s ago, limit is 0.5 s
        time.sleep(0.5)
        assert released
    finally:
        stop.set()
        thread.join(1)
    assert embeddings.start_idle_unloader(0) is None
