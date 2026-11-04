"""The graph is rebuilt on the worker, once per session (plan step 6)."""

from datetime import timedelta

import numpy as np

from zettlecast import graphstate, linker, worker
from zettlecast.config import settings
from zettlecast.db import Database, index_payload
from zettlecast.models import NoteMetadata, NoteModel

from test_worker import fake_vectors, primary  # noqa: F401  (fixtures)

TEXTS = ["Heat training raises plasma volume over ten days of exposure.",
         "Heat acclimation and plasma volume: ten days of training in the heat.",
         "Tapering before a marathon: cut the volume, keep the intensity.",
         "Marathon taper: three weeks of less volume and the same intensity."]


def corpus(db):
    for i, text in enumerate(TEXTS):
        note = NoteModel(uuid=f"n{i}", title=f"Note {i}", source_type="web", source_path=f"https://x/{i}",
                         full_text=text * 3, content_hash=str(i),
                         metadata=NoteMetadata(tags=["heat"] if i < 2 else ["taper"], custom={"summary": "s"}))
        db.upsert_note(note, index=index_payload(note))


def varied_vectors(monkeypatch):
    """Similar texts get similar vectors, so the graph has something to link."""
    def embed(texts):
        out = np.zeros((len(texts), 1024), np.float32)
        for k, t in enumerate(texts):
            out[k, 0 if "eat" in t else 1] = 1.0
            out[k, 2 + len(t) % 7] = 0.3
        return out / np.linalg.norm(out, axis=1, keepdims=True)
    monkeypatch.setattr("zettlecast.embeddings.embed_documents", embed)


def test_the_flag_keeps_changes_made_while_the_worker_computes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    assert graphstate.claim("w1", timedelta(minutes=10)) is None  # up to date: nothing to do
    graphstate.mark_dirty()
    job = graphstate.claim("w1", timedelta(minutes=10))
    assert job and graphstate.claim("w2", timedelta(minutes=10)) is None  # held
    graphstate.mark_dirty()  # a note stored meanwhile
    assert graphstate.finish(job["id"], stored=True)
    assert graphstate.dirty()  # the rebuild didn't include it: another one next time
    job = graphstate.claim("w1", timedelta(minutes=10))
    assert graphstate.finish(job["id"], stored=True) and not graphstate.dirty()


def test_the_graph_job_comes_last_and_matches_a_local_rebuild(primary, tmp_path, monkeypatch):  # noqa: F811
    db = Database(tmp_path / "db")
    for target in ("zettlecast.db.db", "zettlecast.linker.db", "zettlecast.main.db"):
        monkeypatch.setattr(target, db)
    varied_vectors(monkeypatch)
    corpus(db)  # stored indexed on the primary: the graph is out of date
    assert graphstate.dirty() and worker.primary().post("/worker/claim", {"worker_id": "w1", "kinds": worker.KINDS})["kind"] == "graph"

    # what a standalone install would compute from the same database
    expected_edges, expected_communities = linker.compute_graph(
        db.all_note_vectors(), db.all_chunk_vectors(), db.get_all_rejected_pairs(), db.get_communities())
    assert expected_edges

    job = graphstate._load()["job"]
    worker.run_graph(job)  # streams notes + chunks as Arrow, computes, sends back
    stored = sorted((e["source_uuid"], e["target_uuid"], e["kind"]) for e in db.get_all_edges())
    assert stored == sorted((e["source_uuid"], e["target_uuid"], e["kind"]) for e in expected_edges)
    assert len(db.get_communities()) == len(expected_communities)
    assert not graphstate.dirty()
    assert worker.primary().post("/worker/claim", {"worker_id": "w1", "kinds": worker.KINDS})["kind"] is None


def test_work_waiting_comes_before_the_graph(primary, tmp_path, monkeypatch):  # noqa: F811
    from zettlecast.ingestqueue import IngestQueue

    db = Database(tmp_path / "db")
    for target in ("zettlecast.db.db", "zettlecast.main.db"):
        monkeypatch.setattr(target, db)
    graphstate.mark_dirty()
    IngestQueue().add_url("https://example.com/a")
    assert worker.primary().post("/worker/claim", {"worker_id": "w1", "kinds": worker.KINDS})["kind"] == "ingest"


def test_an_older_workers_link_only_marks_the_graph(primary, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(linker, "rebuild_graph", lambda: (_ for _ in ()).throw(AssertionError("rebuilt on the primary")))
    assert worker.primary().post("/worker/link")["status"] == "queued" and graphstate.dirty()
