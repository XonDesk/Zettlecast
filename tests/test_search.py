"""Search keeps other notes in play when one long note matches everywhere."""

from unittest.mock import patch

import numpy as np

from zettlecast import search


def chunks():
    # 60 chunks of one long episode rank above everything from the second note
    rows = [{"chunk_id": f"a{i}", "note_uuid": "a", "text": f"[{i}.0s] x", "start_char": 0, "similarity": 0.9}
            for i in range(60)]
    rows.append({"chunk_id": "b0", "note_uuid": "b", "text": "[5.0s] y", "start_char": 0, "similarity": 0.5})
    return rows


def test_one_long_note_does_not_crowd_out_the_rest():
    seen = {}

    def rerank(query, texts):
        seen["n"] = len(texts)
        return [(i, 0.9 - i * 0.01) for i in range(len(texts))]

    with patch.object(search.db, "search_chunks", return_value=chunks()), \
         patch.object(search.db, "note_titles", return_value={"a": ("A", "audio"), "b": ("B", "audio")}), \
         patch.object(search, "embed_query", return_value=np.zeros(4)), \
         patch.object(search, "rerank", side_effect=rerank):
        results = search.search("q", top_k=5)
    assert [r.uuid for r in results] == ["a", "b"]
    assert seen["n"] == search.MAX_CHUNKS_PER_NOTE + 1


def test_the_primary_ranks_by_similarity_without_the_reranker(monkeypatch):
    from zettlecast.config import settings

    monkeypatch.setattr(settings, "role", "primary")
    assert not settings.use_reranker
    with patch.object(search.db, "search_chunks", return_value=chunks()), \
         patch.object(search.db, "note_titles", return_value={"a": ("A", "audio"), "b": ("B", "audio")}), \
         patch.object(search, "embed_query", return_value=np.zeros(4)), \
         patch.object(search, "rerank", side_effect=AssertionError("reranked on the primary")):
        results = search.search("q", top_k=5)
    assert [r.uuid for r in results] == ["a", "b"] and results[0].score == 0.9
    monkeypatch.setattr(settings, "rerank", True)  # RERANK=true still turns it on
    assert settings.use_reranker


def test_link_reason_names_what_links_two_notes():
    import json

    from zettlecast.search import link_reason

    assert link_reason({"kind": "link"}) == "Linked by you"
    assert link_reason({"kind": "duplicate"}) == "Same content"
    reason = {"shared": [{"kind": "person", "name": "Sophie Herzog"}, {"kind": "topic", "name": "overtraining"}],
              "passage": {"similarity": 0.9}}
    assert link_reason({"kind": "similar", "reason_json": json.dumps(reason)}) == \
        "Shared: Sophie Herzog, overtraining; similar passages"
    assert link_reason({"kind": "similar", "reason_json": "{}"}) == "Similar content"


def test_model_dtype_overrides_the_per_device_choice(monkeypatch):
    import pytest
    import torch

    from zettlecast import embeddings
    from zettlecast.config import Settings, settings

    monkeypatch.setattr(settings, "model_dtype", "float16")  # the primary's container: half of float32
    assert embeddings._dtype() is torch.float16
    with pytest.raises(ValueError):
        Settings(model_dtype="int8")
