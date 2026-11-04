"""Pages and files added on the primary are read by the worker (plan step 5)."""

from datetime import timedelta

import pytest

from zettlecast import tagging, worker
from zettlecast.config import settings
from zettlecast.db import Database, index_payload
from zettlecast.ingestqueue import ERROR, MAX_ATTEMPTS, PLACEHOLDER, IngestQueue
from zettlecast.models import NoteModel, ProcessingResult

from test_worker import as_worker, fake_vectors, primary  # noqa: F401  (fixtures)

PAGE = "Heat training raises plasma volume over ten days; the effect fades within weeks. " * 4


@pytest.fixture
def db(tmp_path, monkeypatch, primary):  # noqa: F811
    database = Database(tmp_path / "db")
    for target in ("zettlecast.db.db", "zettlecast.main.db", "zettlecast.references.db"):
        monkeypatch.setattr(target, database)
    fake_vectors(monkeypatch)
    return database


def parsed(text=PAGE, title="Heat training"):
    return ProcessingResult(status="success", uuid="parsed-uuid", title=title, note=NoteModel(
        uuid="parsed-uuid", title=title, source_type="web", source_path="https://example.com/heat",
        full_text=text, content_hash=f"hash:{text[:20]}"))


def test_a_url_waits_as_a_placeholder_until_the_worker_reads_it(primary, db, monkeypatch):  # noqa: F811
    reply = primary.post("/ingest?url=https://example.com/heat",
                         headers={"Authorization": f"Bearer {settings.api_token}"}).json()
    assert reply["status"] == "queued"
    placeholder = reply["uuid"]
    [listed] = db.list_notes()
    assert listed["uuid"] == placeholder and listed["waiting"]
    assert db.all_note_vectors() == [] and db.chunks.count_rows() == 0
    monkeypatch.setattr(tagging, "_cache", {"at": 0.0, "uuids": None})
    assert tagging._untagged() == []  # not the tagging job's: the worker reads the page itself

    batch = worker.primary().post("/worker/claim", {"worker_id": "w1", "kinds": worker.KINDS})
    assert batch["kind"] == "ingest"
    monkeypatch.setattr("zettlecast.parser.parse_url", lambda url, enrich=True: parsed())
    links = []
    monkeypatch.setattr(worker, "_primary", worker.primary())
    monkeypatch.setattr(worker.primary(), "post", _recording(worker.primary().post, links))
    worker.run_ingest(batch["items"])

    note = db.get_note_by_uuid(placeholder)
    assert note.title == "Heat training" and PLACEHOLDER not in note.metadata.custom
    assert db.chunks.count_rows() > 0 and len(db.all_note_vectors()) == 1 and not db.list_notes()[0]["waiting"]
    assert "/worker/link" in links  # the batch is linked once
    assert IngestQueue().items[batch["items"][0]["id"]]["status"] == "completed"


def _recording(post, calls):
    def wrapped(path, *args, **kwargs):
        calls.append(path)
        return post(path, *args, **kwargs)
    return wrapped


def test_an_uploaded_pdf_stays_on_the_primary_and_the_note_points_at_it(primary, db, tmp_path, monkeypatch):  # noqa: F811
    upload = primary.post("/ingest", files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
                          headers={"Authorization": f"Bearer {settings.api_token}"}).json()
    assert upload["status"] == "queued"
    kept = tmp_path / "uploads" / "paper.pdf"
    [item] = worker.primary().post("/worker/claim", {"worker_id": "w1", "kinds": ["ingest"]})["items"]

    seen = {}

    def parse_file(path):
        seen["bytes"] = path.read_bytes()
        return parsed(title="A paper")

    monkeypatch.setattr("zettlecast.parser.parse_file", parse_file)
    as_worker(monkeypatch, tmp_path / "worker")
    worker.run_ingest([item])
    assert seen["bytes"] == b"%PDF-1.4 fake" and kept.read_bytes() == b"%PDF-1.4 fake"  # downloaded, not moved
    assert db.get_note_by_uuid(upload["uuid"]).source_path == str(kept)


def test_a_page_already_stored_replaces_its_placeholder_with_a_duplicate(primary, db, monkeypatch):  # noqa: F811
    stored = parsed().note
    stored.uuid = "already-here"
    db.upsert_note(stored, index=index_payload(stored))
    item = IngestQueue().add_url("https://example.com/heat-again")
    IngestQueue().claim("w1", timedelta(minutes=10))
    done = IngestQueue().complete(item["id"], parsed().note, None)
    assert done["status"] == "duplicate" and done["note"] == "already-here"
    assert db.get_note_by_uuid(item["placeholder"]) is None


def test_after_its_last_attempt_the_placeholder_says_why(primary, db):  # noqa: F811
    item = IngestQueue().add_url("https://example.com/gone")
    for attempt in range(MAX_ATTEMPTS):
        IngestQueue().claim("w1", timedelta(minutes=10))
        result = IngestQueue().fail(item["id"], "HTTP 404")
    assert result["status"] == "failed"
    note = db.get_note_by_uuid(item["placeholder"])
    assert note.metadata.custom[ERROR] == "HTTP 404" and PLACEHOLDER not in note.metadata.custom
    assert db.list_notes()[0]["add_failed"] and not db.list_notes()[0]["waiting"]


def test_an_old_worker_that_doesnt_know_ingest_isnt_given_it(primary, db):  # noqa: F811
    IngestQueue().add_url("https://example.com/heat")
    assert worker.primary().post("/worker/claim", {"worker_id": "old", "kinds": ["podcast", "image", "tag"]})["kind"] is None
