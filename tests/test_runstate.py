"""Run state for the process page: published progress, stop requests, one run at a time."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from zettlecast import pipeline, runstate
from zettlecast.models import NoteModel
from test_two_pass import FakeTranscriber, make_queue


@pytest.fixture(autouse=True)
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(runstate.settings, "storage_path", tmp_path / "storage")
    return tmp_path / "storage"


def run_queue(tmp_path, queue, run, after_note=None):
    """Two-pass run with fakes; after_note(n) is called as each note is saved."""
    saved = []

    def save_result(res, episode, enhanced):
        note = tmp_path / f"{episode.id}.md"
        note.write_text("x", encoding="utf-8")
        saved.append(episode.id)
        if after_note:
            after_note(len(saved))
        return note

    parsed = type("P", (), {"status": "success", "note": NoteModel(title="t", full_text="x", source_type="audio",
                                                                    source_path="x", content_hash="h")})()
    with patch("zettlecast.podcast.queue.TranscriptionQueue", return_value=queue), \
         patch("zettlecast.podcast.nemo_transcriber.NeMoTranscriber", side_effect=lambda: FakeTranscriber([])), \
         patch("zettlecast.podcast.formatter.save_result", side_effect=save_result), \
         patch.object(pipeline, "parse_file", return_value=parsed), \
         patch.object(pipeline.db, "upsert_note"), \
         patch.object(pipeline, "update_edges_for_note"), \
         patch("zettlecast.llm.unload"):
        return pipeline._run_podcast_queue(enhance=False, sync=False, run=run), saved


def test_run_publishes_phases_stages_and_finished_episodes(tmp_path):
    run = runstate.RunState("cli")
    run_queue(tmp_path, make_queue(tmp_path), run)
    state = runstate.read()
    assert state["phase"] == "enhance" and state["phase_done"] == 2
    assert state["completed"] == 2 and [r["title"] for r in state["recent"]] == ["Episode 1", "Episode 0"]
    assert state["alive"]  # this process is the run
    run.finish()
    assert runstate.read()["phase"] == "idle" and not runstate.read()["alive"]


def test_stop_request_ends_the_run_after_the_current_episode(tmp_path):
    queue = make_queue(tmp_path, n=3)
    run = runstate.RunState("api")
    summary, saved = run_queue(tmp_path, queue, run, after_note=lambda n: runstate.request_stop())
    assert summary.completed == 1 and len(saved) == 1
    statuses = sorted(i.status for i in queue.items.values())
    assert statuses == ["completed", "pending", "pending"]  # the rest went back, not left "processing"
    assert all(i.attempts == 0 for i in queue.items.values() if i.status == "pending")
    assert runstate.read()["stop_requested"]
    run.finish()
    assert not runstate.stop_requested()


def test_only_one_run_at_a_time(monkeypatch):
    runstate.RunState("cli")  # a live run in this process...
    monkeypatch.setattr(runstate, "active_run", lambda: {"pid": os.getpid() + 1, "source": "cli"})
    with pytest.raises(pipeline.RunActiveError):  # ...seen from another process
        pipeline.run_podcast_queue()


def test_dead_run_is_not_alive(storage):
    runstate._write_json(storage / "run_state.json", {"phase": "enhance", "pid": 999999, "started_at": "2026-01-01T00:00:00"})
    assert not runstate.read()["alive"]
    assert runstate.active_run() is None


def test_missing_state_reads_idle():
    assert runstate.read() == {"phase": "idle", "alive": False}


def test_state_file_location(storage):
    runstate.RunState("cli")
    assert Path(storage / "run_state.json").exists()
