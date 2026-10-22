"""
Run state shared between processes.

A podcast run (from the CLI or the API server) publishes what it is doing to
<storage>/run_state.json, so the web UI's process page shows a CLI run too, and reads
<storage>/run_control.json between episodes, so any process can ask it to stop.
Only one run may use the GPU at a time: `active_run()` tells whether one is going.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)

RECENT = 12  # finished episodes kept for the process page
CYCLES = 30  # episode times kept per phase (cycle time, ETA)


def _state_file() -> Path:
    return settings.storage_path / "run_state.json"


def _control_file() -> Path:
    return settings.storage_path / "run_control.json"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    for _ in range(5):  # a reader may hold the file open for a moment on Windows
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    logger.debug(f"Could not update {path.name}")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _seconds_since(iso: Optional[str]) -> float:
    return (datetime.now() - datetime.fromisoformat(iso)).total_seconds() if iso else 0.0


def _previous_averages() -> dict:
    """Seconds per episode of each phase (graph: seconds per rebuild) from earlier runs, for the ETA."""
    try:
        return dict(json.loads(_state_file().read_text(encoding="utf-8")).get("averages") or {})
    except (OSError, ValueError, AttributeError):
        return {}


def pid_alive(pid: Optional[int], since: Optional[str] = None) -> bool:
    """True if the process exists and (with since) was started before that time."""
    if not pid:
        return False
    import psutil

    try:
        proc = psutil.Process(pid)
        if since:
            # a process started after the state was written reuses the pid of one that exited
            return proc.create_time() <= datetime.fromisoformat(since).timestamp() + 2
        return proc.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


def read() -> dict:
    """The last published state, with `alive`: whether its run is still going."""
    if settings.is_primary:  # runs happen on the worker, which reports them with its heartbeat
        from . import workerstate

        return workerstate.run_state()
    try:
        state = json.loads(_state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"phase": "idle", "alive": False}
    state["alive"] = state.get("phase", "idle") != "idle" and pid_alive(state.get("pid"), state.get("started_at"))
    state["stop_requested"] = stop_requested()
    return state


def active_run() -> Optional[dict]:
    """The state of a run that is going (any process), else None."""
    state = read()
    return state if state["alive"] else None


class RunState:
    """Writer for one run (lives in the process doing the run)."""

    def __init__(self, source: str):
        averages = _previous_averages()
        self.state = {
            "pid": os.getpid(),
            "source": source,  # cli | api
            "started_at": _now(),
            "phase": "starting",  # starting | transcribe | enhance | graph | idle
            "phase_total": 0,
            "phase_done": 0,
            "phase_started_at": None,
            "cycles": [],  # seconds per finished episode of this phase, newest last
            "averages": averages,
            "episode_id": None,
            "episode_title": None,
            "episode_started_at": None,
            "stage": None,
            "stage_started_at": None,
            "chunk": 0,
            "chunks": 0,
            "stage_times": {},  # seconds per finished step of the current episode
            "last_stages": {},  # the same for the previous episode
            "completed": 0,
            "failed": 0,
            "recent": [],
            "updated_at": _now(),
        }
        clear_stop()
        self._save()

    def _save(self) -> None:
        self.state["updated_at"] = _now()
        _write_json(_state_file(), self.state)

    def _close_phase(self) -> None:
        """Remember how long the phase that is ending took per episode (graph: per rebuild)."""
        name, cycles = self.state["phase"], self.state["cycles"]
        if name == "graph" and self.state["phase_started_at"]:
            self.state["averages"]["graph"] = round(_seconds_since(self.state["phase_started_at"]), 1)
        elif cycles:
            self.state["averages"][name] = round(sum(cycles) / len(cycles), 1)

    def _close_stage(self) -> None:
        stage = self.state["stage"]
        if stage and self.state["stage_started_at"]:
            times = self.state["stage_times"]
            times[stage] = round(times.get(stage, 0.0) + _seconds_since(self.state["stage_started_at"]), 1)

    def phase(self, name: str, total: int = 0) -> None:
        self._close_phase()
        self.state.update(phase=name, phase_total=total, phase_done=0, phase_started_at=_now(), cycles=[],
                          episode_id=None, episode_title=None, stage=None, chunk=0, chunks=0,
                          stage_times={}, last_stages={})
        self._save()

    def episode(self, episode_id: str, title: str) -> None:
        self.state.update(episode_id=episode_id, episode_title=title, episode_started_at=_now(),
                          stage=None, chunk=0, chunks=0, stage_times={})
        self._save()

    def stage(self, name: str, chunk: int = 0, chunks: int = 0) -> None:
        if name != self.state["stage"]:
            self._close_stage()
            self.state["stage_started_at"] = _now()
        self.state.update(stage=name, chunk=chunk, chunks=chunks)
        self._save()

    def done(self, title: str, status: str, seconds: float = 0.0, audio_seconds: float = 0.0,
             count: bool = True) -> None:
        """An episode finished this phase (count: also a finished episode of the run)."""
        self.state["phase_done"] += 1
        self._close_stage()
        if status == "completed" and seconds:
            self.state["cycles"] = (self.state["cycles"] + [round(seconds, 1)])[-CYCLES:]
            self.state["last_stages"] = self.state["stage_times"]
        self.state["stage_times"] = {}
        if count:
            self.state["completed" if status == "completed" else "failed"] += status != "cancelled"
            self.state["recent"] = ([{
                "title": title, "status": status, "seconds": round(seconds, 1),
                "audio_minutes": round(audio_seconds / 60, 1), "finished_at": _now(),
            }] + self.state["recent"])[:RECENT]
        elif status == "failed":
            self.state["failed"] += 1
        self.state.update(episode_id=None, episode_title=None, stage=None, chunk=0, chunks=0)
        self._save()

    def finish(self) -> None:
        self._close_phase()
        self.state.update(phase="idle", episode_id=None, episode_title=None, stage=None,
                          finished_at=_now())
        clear_stop()
        self._save()


def mark_aborted() -> None:
    """Close out the state of a run whose process was ended from outside."""
    try:
        state = json.loads(_state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    state.update(phase="idle", episode_id=None, episode_title=None, stage=None,
                 finished_at=_now(), aborted=True)
    _write_json(_state_file(), state)
    clear_stop()


# --- Stop requests ---


def request_stop() -> None:
    """Ask the running run (any process) to stop after its current episode."""
    _write_json(_control_file(), {"stop": "after_episode", "requested_at": _now()})


def stop_requested() -> bool:
    try:
        return json.loads(_control_file().read_text(encoding="utf-8")).get("stop") == "after_episode"
    except (OSError, ValueError):
        return False


def clear_stop() -> None:
    try:
        _control_file().unlink()
    except FileNotFoundError:
        pass
