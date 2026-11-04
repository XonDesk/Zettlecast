"""Database maintenance."""

from datetime import timedelta

import numpy as np

from zettlecast.db import DIM, Database
from zettlecast.models import NoteModel


def test_optimize_drops_old_versions_and_keeps_the_data(tmp_path, monkeypatch):
    monkeypatch.setattr("zettlecast.embeddings.embed_documents",
                        lambda texts: np.ones((len(texts), DIM), np.float32))
    db = Database(tmp_path / "db")
    for i in range(5):
        db.upsert_note(NoteModel(title=f"Note {i}", full_text=f"Some text number {i}.",
                                 source_type="markdown", source_path=f"note{i}.md", content_hash=str(i)))
    before = len(db.notes.list_versions())
    db.optimize(keep=timedelta(0))
    assert len(db.notes.list_versions()) < before
    assert db.notes.count_rows() == 5
    assert db.chunks.count_rows() == 5


# --- The primary stores notes waiting for the worker's index (plan step 4) ---

import pytest

from zettlecast.config import settings
from zettlecast.db import INDEX_PENDING, index_payload


def note(uuid="n1", text="Heat training raises plasma volume over ten days of exposure."):
    return NoteModel(uuid=uuid, title="Heat", full_text=text, source_type="web", source_path="https://x",
                     content_hash="h")


@pytest.fixture
def vectors(monkeypatch):
    """The worker's embedder: no model loaded."""
    monkeypatch.setattr("zettlecast.embeddings.embed_documents",
                        lambda texts: np.ones((len(texts), DIM), np.float32) / np.sqrt(DIM))


def test_the_primary_stores_a_new_note_waiting_without_embedding_it(tmp_path, monkeypatch, vectors):
    db = Database(tmp_path / "db")
    payload = index_payload(note())  # made "on the worker", before the primary stops embedding
    monkeypatch.setattr(settings, "role", "primary")
    monkeypatch.setattr("zettlecast.embeddings.embed_documents", lambda texts: pytest.fail("embedded on the primary"))

    assert db.upsert_note(note()) is False
    stored = db.get_note_by_uuid("n1")
    assert stored.metadata.custom[INDEX_PENDING] is True
    assert db.chunks.count_rows() == 0 and db.all_note_vectors() == []  # not searchable or linked yet
    assert db.list_notes()[0]["waiting"] is True  # but listed

    assert db.upsert_note(stored, index=payload) is True  # the worker's index arrives
    assert INDEX_PENDING not in db.get_note_by_uuid("n1").metadata.custom
    assert db.chunks.count_rows() == 1 and len(db.all_note_vectors()) == 1
    assert db.list_notes()[0]["waiting"] is False


def test_a_changed_note_keeps_its_old_index_until_the_new_one_comes(tmp_path, monkeypatch, vectors):
    db = Database(tmp_path / "db")
    db.upsert_note(note())  # indexed (standalone, or by an earlier worker batch)
    monkeypatch.setattr(settings, "role", "primary")
    assert db.upsert_note(note(text="Heat training, now with the comments' takeaways.")) is False
    assert db.chunks.count_rows() == 1 and len(db.all_note_vectors()) == 1  # still searchable, still linked
    assert db.get_note_by_uuid("n1").full_text.startswith("Heat training, now")


def test_ingest_on_the_primary_does_not_link_a_waiting_note(monkeypatch):
    from zettlecast import pipeline

    monkeypatch.setattr(pipeline.db, "upsert_note", lambda n, index=None: False)
    monkeypatch.setattr(pipeline, "update_edges_for_note", lambda uuid: pytest.fail("linked without an index"))
    pipeline.ingest_note(note())


# --- Fewer versions per stored note (plan step 8) ---


def test_a_stored_note_costs_one_version_per_table_and_drops_chunks_it_no_longer_has(tmp_path, vectors):
    db = Database(tmp_path / "db")
    db.upsert_note(note(text="First passage about heat. " * 80))  # several chunks
    many = db.chunks.count_rows()
    assert many > 1
    notes_v, chunks_v = len(db.notes.list_versions()), len(db.chunks.list_versions())
    db.upsert_note(note(text="Now one short passage."))
    assert len(db.notes.list_versions()) == notes_v + 1 and len(db.chunks.list_versions()) == chunks_v + 1
    assert db.chunks.count_rows() == 1 and db.notes.count_rows() == 1
    assert db.get_note_by_uuid("n1").full_text == "Now one short passage."


def test_the_primary_compacts_in_the_background_after_many_writes(tmp_path, monkeypatch, vectors):
    import zettlecast.db as dbm

    db = Database(tmp_path / "db")
    monkeypatch.setattr(settings, "role", "primary")
    monkeypatch.setattr(dbm, "COMPACT_AFTER_WRITES", 3)
    compacted = []
    monkeypatch.setattr(db, "optimize", lambda: compacted.append(1))
    for i in range(7):
        db.upsert_note(note(uuid=f"n{i}"))  # waiting notes count too
    import time
    time.sleep(0.2)
    assert len(compacted) == 2
