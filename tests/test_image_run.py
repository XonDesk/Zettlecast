"""The image run keeps LLM_PARALLEL reads in flight, stores each result on its own thread, links once."""

import threading
import time
from types import SimpleNamespace

from zettlecast import main
from zettlecast.config import settings


class FakeQueue:
    def __init__(self, n):
        self.items = {str(i): SimpleNamespace(status="pending", image=SimpleNamespace(
            id=str(i), image_path=f"shot{i}.png", image_title=f"shot{i}", keep_image=False, focus_region=None))
            for i in range(n)}

    def get_pending_count(self):
        return sum(i.status == "pending" for i in self.items.values())

    def get_next_pending(self):
        return next((i for i in self.items.values() if i.status == "pending"), None)

    def mark_started(self, id):
        self.items[id].status = "processing"

    def mark_completed(self, id, seconds):
        self.items[id].status = "completed"

    def mark_failed(self, id, error):
        self.items[id].status = "failed"


def test_reads_images_in_parallel_and_stores_each_once(monkeypatch):
    queue = FakeQueue(9)
    monkeypatch.setattr("zettlecast.image.queue.ImageQueue", lambda: queue)
    monkeypatch.setattr(settings, "llm_parallel", 3)
    lock, active, peak, stored = threading.Lock(), [0], [0], []

    def parse(path, **_):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        if path.name == "shot4.png":
            return SimpleNamespace(status="failed", note=None, error_message="unreadable")
        return SimpleNamespace(status="success", note=path.name, error_message=None)

    monkeypatch.setattr("zettlecast.image.image_parser.parse_image", parse)
    monkeypatch.setattr(main.db, "upsert_note", lambda note: stored.append((note, threading.current_thread().name)))
    rebuilds = []
    monkeypatch.setattr("zettlecast.linker.rebuild_graph", lambda: rebuilds.append(len(stored)))
    main._run_image_processing_background()

    assert peak[0] == 3
    assert sorted(n for n, _ in stored) == sorted(f"shot{i}.png" for i in range(9) if i != 4)
    assert {t for _, t in stored} == {threading.current_thread().name}
    assert rebuilds == [8]  # linked once, after all of them
    assert queue.items["4"].status == "failed"
    assert all(i.status == "completed" for k, i in queue.items.items() if k != "4")


def test_stop_finishes_the_images_being_read_and_still_links_them(monkeypatch):
    from zettlecast import system

    queue = FakeQueue(10)
    monkeypatch.setattr("zettlecast.image.queue.ImageQueue", lambda: queue)
    monkeypatch.setattr(settings, "llm_parallel", 2)
    stored, rebuilds = [], []

    def parse(path, **_):
        if path.name == "shot1.png":
            system.image_run["stop_requested"] = True  # STOP pressed while the first two are read
        time.sleep(0.02)
        return SimpleNamespace(status="success", note=path.name, error_message=None)

    monkeypatch.setattr("zettlecast.image.image_parser.parse_image", parse)
    monkeypatch.setattr(main.db, "upsert_note", stored.append)
    monkeypatch.setattr("zettlecast.linker.rebuild_graph", lambda: rebuilds.append(len(stored)))
    main._run_image_processing_background()

    assert sorted(stored) == ["shot0.png", "shot1.png"]  # the two in flight, no more
    assert rebuilds == [2]
    assert sum(i.status == "pending" for i in queue.items.values()) == 8
    assert not system.image_run["is_running"] and system.image_run["phase"] is None
