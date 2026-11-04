"""Two machines: the primary's job API, the worker's side of it, leases, wake-on-LAN, idle shutdown."""

import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from zettlecast import db as db_module
from zettlecast import main, migrate, paths, statefile, tagging, wake, worker, workerstate
from zettlecast.config import settings
from zettlecast.image.queue import ImageQueue
from zettlecast.models import NoteMetadata, NoteModel
from zettlecast.podcast.queue import TranscriptionQueue

TOKEN = "worker-secret"


@pytest.fixture
def primary(tmp_path, monkeypatch):
    """This process as the primary, storage in tmp_path; a worker client talking to it."""
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    monkeypatch.setattr(settings, "role", "primary")
    monkeypatch.setattr(settings, "worker_token", TOKEN)
    monkeypatch.setattr(workerstate, "_beat", {"at": None, "seen_at": None, "data": {}})
    monkeypatch.setattr(workerstate, "_commands", {"stop": False, "abort": False, "cancel_episode": None,
                                                   "image_stop": False})
    monkeypatch.setattr(tagging, "_claimed", {})
    monkeypatch.setattr(tagging, "pending_count", lambda: 0)
    client = TestClient(main.app)
    monkeypatch.setattr(worker, "_primary", worker.Primary(url="http://testserver", token=TOKEN, client=client))
    return client


def as_worker(monkeypatch, root: Path) -> None:
    """The worker's side of this process uses its own storage folder (the primary's stays)."""
    own = SimpleNamespace(storage_path=root)
    monkeypatch.setattr(worker, "settings", own)
    monkeypatch.setattr(paths, "settings", own)


def held_audio(tmp_path, name="episode.mp3", show="Show") -> Path:
    path = tmp_path / "podcasts" / show / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ID3 fake audio " + name.encode())
    return path


# --- State files ---


def test_merge_keeps_changes_made_by_another_writer(tmp_path):
    path = tmp_path / "q.json"
    statefile.write(path, {"items": {"a": {"status": "pending"}, "b": {"status": "pending"}}, "times": [1]})
    base = statefile.read(path)
    # another writer finishes "b" and records a time meanwhile
    statefile.write(path, {"items": {"a": {"status": "pending"}, "b": {"status": "completed"}}, "times": [1, 2]})
    mine = {"items": {"a": {"status": "processing"}, "b": {"status": "pending"}, "c": {"status": "pending"}},
            "times": [1, 3]}
    merged = statefile.save(path, base, mine, list_keys={"times": 10})
    assert merged["items"] == {"a": {"status": "processing"}, "b": {"status": "completed"}, "c": {"status": "pending"}}
    assert merged["times"] == [1, 2, 3]
    assert statefile.read(path) == merged


def test_two_queue_instances_dont_undo_each_other(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    first = held_audio(tmp_path, "one.mp3")
    second = held_audio(tmp_path, "two.mp3")
    q = TranscriptionQueue()
    a = q.add(first, podcast_name="Show")
    importer = TranscriptionQueue()  # loaded before the other change, like a long feed import
    TranscriptionQueue().mark_started(a)
    importer.add(second, podcast_name="Show")
    statuses = {i.episode.episode_title: i.status for i in TranscriptionQueue().items.values()}
    assert statuses == {"one": "processing", "two": "pending"}


# --- Leases ---


def test_lease_runs_out_and_renews(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    q = TranscriptionQueue()
    job = q.add(held_audio(tmp_path), podcast_name="Show")
    [item] = q.claim("w1", timedelta(minutes=10))
    assert item.status == "processing" and item.worker_id == "w1" and item.worker_pid is None
    assert TranscriptionQueue().items[job].status == "processing"  # held, though no local process has it

    assert TranscriptionQueue().renew("w1", timedelta(seconds=-1)) == 1  # lapsed
    assert TranscriptionQueue().items[job].status == "pending"  # loading resets it
    assert TranscriptionQueue().items[job].worker_id is None


def test_image_claims_count_attempts_and_release_returns_them(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    q = ImageQueue()
    job = q.add_remote("desktop:C:/shots/a.png", "h" * 64, title="a", width=100, height=200)
    assert q.add_remote("desktop:C:/shots/a.png", "h" * 64) == job  # same image: same job
    [item] = ImageQueue().claim("w1", timedelta(minutes=5))
    assert item.attempts == 1 and item.image.megapixels == pytest.approx(0.02)
    assert ImageQueue().release([job]) == 1
    item = ImageQueue().items[job]
    assert (item.status, item.attempts, item.worker_id) == ("pending", 0, None)


# --- Paths ---


def test_worker_refs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    inside = tmp_path / "podcasts" / "Show" / "ep.mp3"
    assert paths.to_worker_ref(inside) == "desktop:podcasts/Show/ep.mp3"
    assert paths.from_worker_ref("desktop:podcasts/Show/ep.mp3") == inside
    outside = Path("C:/Users/me/Pictures/shot.png")
    assert paths.from_worker_ref("desktop:C:/Users/me/Pictures/shot.png") == outside
    with pytest.raises(ValueError):
        paths.from_worker_ref("desktop:../secrets.txt")


def test_migrate_maps_copied_files_locally_and_the_rest_to_the_desktop(tmp_path):
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "ep_1234abcd.md").write_text("x")
    root = "C:\\Users\\me\\_BRAIN_STORAGE"
    assert migrate.map_path(root + "\\transcripts\\ep_1234abcd.md", root, tmp_path) == \
        str(tmp_path / "transcripts" / "ep_1234abcd.md")
    assert migrate.map_path(root + "\\podcasts\\Show\\ep.mp3", root, tmp_path) == "desktop:podcasts/Show/ep.mp3"
    assert migrate.map_path("C:\\Users\\me\\OneDrive\\Shots\\a.png", root, tmp_path) == \
        "desktop:C:/Users/me/OneDrive/Shots/a.png"
    for unchanged in ("https://example.com/a", "desktop:podcasts/x.mp3", "/Users/me/x.md", None):
        assert migrate.map_path(unchanged, root, tmp_path) is None


# --- The job API ---


def test_worker_api_needs_the_worker_token(primary):
    anonymous = TestClient(main.app)  # the fixture's client carries the worker token
    assert anonymous.post("/worker/claim", json={"worker_id": "w"}).status_code == 401
    assert anonymous.post("/worker/claim", json={"worker_id": "w"},
                          headers={"Authorization": "Bearer nope"}).status_code == 401
    assert anonymous.post(f"/worker/claim?token={settings.api_token}", json={"worker_id": "w"}).status_code == 401


def fake_vectors(monkeypatch):
    """The worker's embedder: one recognisable vector per text, no model loaded."""
    import numpy as np

    def embed(texts):
        vectors = np.zeros((len(texts), db_module.DIM), np.float32)
        vectors[:, 0] = 1.0
        return vectors

    monkeypatch.setattr("zettlecast.embeddings.embed_documents", embed)


def test_index_that_doesnt_fit_the_note_is_not_used(monkeypatch):
    fake_vectors(monkeypatch)
    note = NoteModel(uuid="n1", title="t", source_type="web", source_path="https://x",
                     full_text="Heat training raises plasma volume. " * 10, content_hash="h")
    index = db_module.index_payload(note)
    assert db_module._from_payload(note, index)[0] is not None
    changed = note.model_copy(update={"full_text": note.full_text + " More."})
    assert db_module._from_payload(changed, index) == (None, None)  # text changed since: embed it again
    assert db_module._from_payload(note, {**index, "dim": 3}) == (None, None)


def test_podcast_round_trip(primary, tmp_path, monkeypatch):
    """Claim a held upload, fetch it, send the note: stored, completed, primary's copy gone."""
    audio = held_audio(tmp_path, "upload.mp3", show="Uploads")
    original = audio.read_bytes()
    job = TranscriptionQueue().add(audio, podcast_name="Uploads")
    stored = []
    monkeypatch.setattr("zettlecast.db.db.upsert_note", lambda note, index=None: stored.append((note, index)))
    monkeypatch.setattr("zettlecast.pipeline.update_edges_for_note", lambda uuid: pytest.fail("linked per note"))
    fake_vectors(monkeypatch)

    batch = worker.primary().post("/worker/claim", {"worker_id": "w1"})
    assert batch["kind"] == "podcast"
    [entry] = batch["items"]
    assert entry["held"] and entry["rel"] == "podcasts/Uploads/upload.mp3"

    # the worker's storage is elsewhere: the input is downloaded there and the job points at it
    worker_root = tmp_path / "worker"
    as_worker(monkeypatch, worker_root)
    local = worker.local_input("podcast", entry, entry["episode"]["audio_path"])
    assert local == worker_root / "podcasts" / "Uploads" / "upload.mp3" and local.read_bytes() == original
    assert not audio.exists()  # the primary's copy is deleted once fetched
    assert TranscriptionQueue().items[job].episode.audio_path == "desktop:podcasts/Uploads/upload.mp3"

    note = tmp_path / "out" / "Upload_1234abcd.md"
    note.parent.mkdir()
    note.write_text(f"---\nuuid: {job}\ntitle: Upload\nsource: {local}\n---\n\n## Transcript\nhello\n",
                    encoding="utf-8")
    queue = worker.RemoteTranscriptionQueue([])
    queue.items[job] = SimpleNamespace(status="pending")
    queue.mark_completed(job, SimpleNamespace(processing_time_seconds=12.5), note)

    saved = tmp_path / "transcripts" / "Upload_1234abcd.md"
    assert "source: desktop:podcasts/Uploads/upload.mp3" in saved.read_text(encoding="utf-8")
    # the primary read the note back itself and takes the worker's vectors as they are
    [(stored_note, index)] = stored
    monkeypatch.setattr("zettlecast.embeddings.embed_documents", lambda texts: pytest.fail("embedded on the primary"))
    chunks, vectors = db_module._from_payload(stored_note, index)
    assert chunks and vectors.shape == (len(chunks), db_module.DIM) and vectors[0][0] == 1.0
    item = TranscriptionQueue().items[job]
    assert item.status == "completed" and item.worker_id is None


def test_failed_and_released_episodes(primary, tmp_path):
    q = TranscriptionQueue()
    a = q.add(held_audio(tmp_path, "a.mp3"), podcast_name="Show")
    b = q.add(held_audio(tmp_path, "b.mp3"), podcast_name="Show")
    worker.primary().post("/worker/claim", {"worker_id": "w1"})
    remote = worker.RemoteTranscriptionQueue([])
    remote.items = {a: SimpleNamespace(status="pending"), b: SimpleNamespace(status="pending")}
    remote.mark_started(a)
    remote.mark_failed(a, "boom")
    assert remote.release([b]) == 1
    items = TranscriptionQueue().items
    assert (items[a].status, items[a].error_message, items[a].attempts) == ("pending", "boom", 1)
    assert items[b].status == "pending"


def test_paused_primary_hands_out_nothing(primary, tmp_path):
    TranscriptionQueue().add(held_audio(tmp_path), podcast_name="Show")
    workerstate.request_stop()
    assert worker.primary().post("/worker/claim", {"worker_id": "w1"}) == {"kind": None, "paused": True}
    workerstate.resume()
    assert worker.primary().post("/worker/claim", {"worker_id": "w1"})["kind"] == "podcast"


def test_triage_keep_and_image_result(primary, tmp_path, monkeypatch):
    stored = []
    monkeypatch.setattr("zettlecast.db.db.upsert_note", lambda note, index=None: stored.append(note))
    reply = worker.primary().post("/worker/image/add", {
        "ref": "desktop:C:/Shots/a.png", "image_hash": "a" * 64, "title": "a", "keep_image": True})
    job = reply["job_id"]
    batch = worker.primary().post("/worker/claim", {"worker_id": "w1", "kinds": ["image"]})
    assert batch["kind"] == "image" and not batch["items"][0]["held"]

    kept = tmp_path / "worker" / "images" / "abc.png"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"\x89PNG fake")
    as_worker(monkeypatch, tmp_path / "worker")
    note = NoteModel(uuid="n1", title="Book: Dune", source_type="image", source_path=str(kept), full_text="x",
                     content_hash="h", metadata=NoteMetadata(custom={"image_kept": True,
                                                                     "original_path": "C:/Shots/a.png"}))
    payload = worker.image_for_primary(note)
    worker.primary().post(f"/worker/image/{job}/completed", {**payload, "seconds": 3.0})

    [note] = stored
    assert Path(note.source_path).parent == tmp_path / "images" and Path(note.source_path).read_bytes() == b"\x89PNG fake"
    assert note.metadata.custom["original_path"] == "desktop:C:/Shots/a.png"
    assert ImageQueue().items[job].status == "completed"
    # an undo after it was read keeps the note
    assert worker.primary().post("/worker/image/remove", {"image_hash": "a" * 64}) == {"removed": False, "processed": True}


def test_heartbeat_renews_leases_and_carries_commands(primary, tmp_path):
    job = TranscriptionQueue().add(held_audio(tmp_path), podcast_name="Show")
    [claimed] = TranscriptionQueue().claim("w1", timedelta(seconds=30))
    reply = worker.primary().post("/worker/heartbeat", {"worker_id": "w1", "busy": True,
                                                         "run": {"phase": "transcribe", "episode_id": job}})
    assert reply["stop"] is False and reply["shutdown_ok"] is True
    assert TranscriptionQueue().items[job].lease_until > claimed.lease_until + timedelta(minutes=5)
    assert workerstate.online() and workerstate.run_state()["alive"]

    assert workerstate.cancel_episode(job)
    workerstate.request_stop("abort")
    reply = worker.primary().post("/worker/heartbeat", {"worker_id": "w1", "busy": True,
                                                         "run": {"phase": "transcribe", "episode_id": job}})
    assert reply["abort"] and reply["stop"] and reply["cancel_episode"] == job and reply["paused"]
    # the run ended: stop / abort are done with, the pause stays until Resume
    reply = worker.primary().post("/worker/heartbeat", {"worker_id": "w1", "busy": False, "run": {"phase": "idle"}})
    assert not reply["stop"] and not reply["abort"] and reply["paused"]
    workerstate.set_stay_on(True)
    assert worker.primary().post("/worker/heartbeat", {"worker_id": "w1"})["shutdown_ok"] is False


def test_worker_applies_stop_to_its_local_run(tmp_path, monkeypatch):
    from zettlecast import runstate

    monkeypatch.setattr(settings, "storage_path", tmp_path)
    monkeypatch.setattr(settings, "role", "worker")
    run = runstate.RunState("worker")
    run.phase("transcribe", 2)
    worker.apply_commands({"stop": True, "abort": False, "paused": True})
    assert runstate.stop_requested()
    assert worker._cancel_check("ep1")("asr", 1, 2, "cuda") is True
    worker.apply_commands({"stop": True, "abort": True, "paused": True})
    assert worker._cancel_check("ep1")("asr", 1, 2, "cuda") is False
    run.finish()
    worker.apply_commands({"stop": False, "abort": False, "paused": False})


# --- Tagging ---


def test_what_needs_the_llm():
    long_text = "word " * 50
    assert tagging.needs_llm("web", {"custom": {}}, long_text)
    assert not tagging.needs_llm("web", {"custom": {"summary": "s"}}, long_text)
    assert not tagging.needs_llm("web", {"custom": {}}, "short")
    assert not tagging.needs_llm("audio", {"custom": {}}, long_text)
    assert tagging.needs_llm("reddit", {"custom": {"summary": "s", "reddit": {"thread_fetched": True}}}, "")
    assert not tagging.needs_llm("reddit", {"custom": {"summary": "s", "reddit": {"thread_fetched": True,
                                                                                    "curated": True}}}, "")


def test_title_only_saves_dont_look_like_the_llm_being_down(monkeypatch):
    """Three in a row used to fail ("no usable answer") and cut every batch short."""
    monkeypatch.setattr(settings, "enable_auto_tagging", True)
    monkeypatch.setattr("zettlecast.enricher.chat", lambda *a, **k: pytest.fail("the LLM was asked about a title"))
    for n in range(3):
        note = NoteModel(uuid=f"r{n}", title="Floating Bed (Kinda)", source_type="reddit",
                         source_path="https://reddit.com/x", full_text="r/diy · posted by u/x\n\nLink: https://i.redd.it/b.jpg",
                         content_hash="h", metadata=NoteMetadata(custom={"reddit": {}}))
        done, error = tagging.llm_pass(note)
        assert done is not None and error == ""


def test_a_tagged_note_comes_back_with_its_vectors(primary, monkeypatch):
    fake_vectors(monkeypatch)
    note = NoteModel(uuid="t1", title="Heat", source_type="web", source_path="https://x",
                     full_text="Heat training raises plasma volume. " * 10, content_hash="h")
    monkeypatch.setattr("zettlecast.db.db.get_note_by_uuid", lambda uuid: note)
    stored = []
    monkeypatch.setattr("zettlecast.db.db.upsert_note", lambda n, index=None: stored.append(index))
    worker.primary().post("/worker/tag/t1", {"note": note.model_dump(mode="json"),
                                             "index": db_module.index_payload(note)})
    [index] = stored
    assert db_module._from_payload(note, index)[0] is not None


def test_tag_failures_are_counted_then_given_up(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    monkeypatch.setattr(settings, "enable_auto_tagging", True)
    monkeypatch.setattr(tagging, "_claimed", {})
    monkeypatch.setattr(tagging, "_untagged", lambda: ["u1"])
    monkeypatch.setattr("zettlecast.db.db.get_note_by_uuid",
                        lambda uuid: NoteModel(uuid=uuid, title="t", source_type="web", source_path="https://x",
                                               full_text="x", content_hash="h"))
    for _ in range(tagging.MAX_ATTEMPTS):
        [note] = tagging.claim(5, 60)
        assert tagging.claim(5, 60) == []  # held while the worker has it
        tagging.failed(note.uuid, "LLM down")
    assert tagging.claim(5, 60) == [] and tagging.pending_count() == 0


def test_primary_never_calls_the_llm(monkeypatch):
    from zettlecast import llm
    from zettlecast.enricher import enrich_note

    monkeypatch.setattr(settings, "role", "primary")
    with pytest.raises(llm.LLMError):
        llm.chat("hello")
    note = NoteModel(title="t", source_type="web", source_path="https://x", full_text="x " * 100, content_hash="h")
    assert "summary" not in enrich_note(note).metadata.custom


# --- Wake-on-LAN ---


def test_magic_packet():
    packet = wake.magic_packet("AA:bb-cc:dd:ee:ff")
    assert packet[:6] == b"\xff" * 6 and packet[6:] == bytes.fromhex("aabbccddeeff") * 16
    with pytest.raises(ValueError):
        wake.magic_packet("not a mac")


@pytest.mark.parametrize("window,hour,expected", [
    ("", 3, True), ("01:00-07:00", 3, True), ("01:00-07:00", 12, False),
    ("22:00-06:00", 23, True), ("22:00-06:00", 5, True), ("22:00-06:00", 12, False),
])
def test_wake_window(window, hour, expected):
    assert wake.in_window(window, datetime(2026, 10, 21, hour, 30)) is expected


def test_wake_policy_sends_retries_and_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    monkeypatch.setattr(settings, "desktop_mac_address", "aa:bb:cc:dd:ee:ff")
    monkeypatch.setattr(settings, "wake_window", "")
    monkeypatch.setattr(workerstate, "_beat", {"at": None, "seen_at": None, "data": {}})
    monkeypatch.setattr(workerstate, "pending_work", lambda: {"podcasts": 2, "images": 0, "tags": 0, "total": 2})
    sent = []
    monkeypatch.setattr(wake, "send", lambda *a, **k: sent.append(1))
    t0 = datetime(2026, 10, 21, 3, 0)

    def at(minutes):
        moment = t0 + timedelta(minutes=minutes)
        monkeypatch.setattr(wake, "datetime", SimpleNamespace(now=lambda: moment, fromisoformat=datetime.fromisoformat))
        return wake.tick(moment)

    assert at(0) == "sent"
    assert at(5) is None  # still booting
    assert at(16) == "sent" and at(32) == "sent"
    assert at(48) == "gave_up" and len(sent) == 3
    assert at(70) is None  # until it's seen or woken by hand
    workerstate.heartbeat({"worker_id": "w1"})  # it came up after all
    assert not workerstate.load()["wake"]["gave_up"]


def test_no_wake_when_paused_or_nothing_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    monkeypatch.setattr(settings, "desktop_mac_address", "aa:bb:cc:dd:ee:ff")
    monkeypatch.setattr(workerstate, "_beat", {"at": None, "seen_at": None, "data": {}})
    monkeypatch.setattr(wake, "send", lambda *a, **k: pytest.fail("woke"))
    monkeypatch.setattr(workerstate, "pending_work", lambda: {"total": 0})
    assert wake.tick() is None
    monkeypatch.setattr(workerstate, "pending_work", lambda: {"podcasts": 1, "total": 1})
    workerstate.update(paused=True)
    assert wake.tick() is None


# --- Idle shutdown ---


def test_idle_shutdown_only_when_nobody_is_signed_in(monkeypatch):
    calls, users = [], [0]
    idle = worker.IdleShutdown(15, users=lambda: users[0], run=calls.append)
    monkeypatch.setitem(worker.commands, "shutdown_ok", True)
    assert not idle.check()  # not idle long enough
    idle.idle_since -= 16 * 60
    users[0] = 1
    assert not idle.check() and not calls  # someone is signed in (console or Remote Desktop)
    users[0] = 0
    assert idle.check() and calls[0][:2] == ["shutdown", "/s"]
    users[0] = 1  # signed in during the grace period: cancelled
    idle.check()
    assert calls[-1] == ["shutdown", "/a"] and not idle.pending
    users[0] = 0
    monkeypatch.setitem(worker.commands, "shutdown_ok", False)  # "stay on" on the process page
    assert not idle.check()


def test_in_a_container_the_wake_goes_to_the_native_helper(tmp_path, monkeypatch):
    import importlib.util

    monkeypatch.setattr(settings, "storage_path", tmp_path)
    monkeypatch.setattr(settings, "wake_sender", "helper")
    monkeypatch.setattr(wake.socket, "socket", lambda *a: pytest.fail("broadcast from the container"))
    wake.send("aa:bb:cc:dd:ee:ff", "192.168.8.255")
    [request] = list(wake.requests_dir().glob("*.json"))
    assert not list(wake.requests_dir().glob(".*.tmp"))

    spec = importlib.util.spec_from_file_location(
        "wake_helper", Path(__file__).parent.parent / "deploy" / "macos" / "wake-helper.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    sent = []
    (wake.requests_dir() / "broken.json").write_text("{", encoding="utf-8")
    assert helper.process(wake.requests_dir(), sender=lambda *a: sent.append(a)) == 1
    assert sent == [("aa:bb:cc:dd:ee:ff", "192.168.8.255", 9)]
    assert not list(wake.requests_dir().iterdir())  # empty: launchd stops running it
    # a request older than MAX_AGE_SECONDS is dropped, not sent late
    wake.send("aa:bb:cc:dd:ee:ff", "192.168.8.255")
    assert helper.process(wake.requests_dir(), now=time.time() + 3600, sender=lambda *a: sent.append(a)) == 0
    assert len(sent) == 1 and not list(wake.requests_dir().iterdir())
