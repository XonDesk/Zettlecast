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
