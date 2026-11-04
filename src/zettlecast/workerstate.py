"""
The worker machine as the primary sees it (ROLE=primary).

The worker sends a heartbeat every WORKER_HEARTBEAT_SECONDS: its run state (the process page's
run), image run, GPU, LLM server, triage counts. The reply carries the commands waiting for it:
stop after the current episode, abort (cancel the episode in progress), cancel one episode,
stop the image run, and whether it may shut its PC down when idle.

The user's switches persist in <storage>/worker_state.json: paused (Stop on the process page:
the worker claims nothing new until Resume) and stay on (no idle shutdown), with the
Wake-on-LAN bookkeeping (wake.py). The last heartbeat is kept in memory.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from . import statefile
from .config import settings

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_beat: dict = {"at": None, "seen_at": None, "data": {}}  # at: time.monotonic() of the last heartbeat
_commands: dict = {"stop": False, "abort": False, "cancel_episode": None, "image_stop": False}


def _offline_after() -> float:
    return max(30.0, 3 * settings.worker_heartbeat_seconds)


def lease() -> timedelta:
    return timedelta(minutes=settings.worker_lease_minutes)


def _path() -> Path:
    return settings.storage_path / "worker_state.json"


def load() -> dict:
    data = statefile.read(_path())
    wake = {"sent_at": None, "misses": 0, "gave_up": False, "reason": None, **(data.get("wake") or {})}
    return {"paused": bool(data.get("paused")), "stay_on": bool(data.get("stay_on")),
            "last_seen_at": data.get("last_seen_at"), "wake": wake}


def save(state: dict) -> None:
    statefile.write(_path(), state)


def update(**changes) -> dict:
    with _lock:
        state = load()
        state.update(changes)
        save(state)
        return state


def update_wake(**changes) -> dict:
    with _lock:
        state = load()
        state["wake"].update(changes)
        save(state)
        return state


def online() -> bool:
    at = _beat["at"]
    return at is not None and time.monotonic() - at < _offline_after()


def last_beat() -> dict:
    """The worker's last heartbeat ({} if none since this process started)."""
    return _beat["data"] if online() else {}


def worker_url() -> str:
    """The worker's file server: WORKER_URL, or the address its heartbeat came from."""
    return settings.worker_url.rstrip("/") or _beat["data"].get("file_server", "")


# --- Heartbeat ---


def heartbeat(data: dict, remote_host: Optional[str] = None) -> dict:
    """Record a heartbeat, renew the worker's leases, and return the commands waiting for it."""
    from . import graphstate, insights, tagging
    from .image.queue import ImageQueue
    from .ingestqueue import IngestQueue
    from .podcast.queue import TranscriptionQueue

    worker_id = data.get("worker_id") or "worker"
    if remote_host and data.get("file_port") and not data.get("file_server"):
        data["file_server"] = f"http://{remote_host}:{data['file_port']}"
    with _lock:
        was_online = online()
        _beat.update(at=time.monotonic(), seen_at=datetime.now().isoformat(timespec="seconds"), data=data)
        state = load()
        if not was_online or state["wake"]["sent_at"] or state["wake"]["misses"]:
            if not was_online:
                logger.info(f"Worker online: {worker_id}")
            state["wake"].update(sent_at=None, misses=0, gave_up=False)
        state["last_seen_at"] = _beat["seen_at"]
        save(state)
        run = data.get("run") or {}
        if not data.get("busy") and run.get("phase", "idle") == "idle":
            # nothing running: stop / abort / cancel have taken effect (paused stays until Resume)
            _commands.update(stop=False, abort=False, cancel_episode=None)
        if not (data.get("images") or {}).get("is_running"):
            _commands["image_stop"] = False
        commands = dict(_commands)
    TranscriptionQueue().renew(worker_id, lease())
    ImageQueue().renew(worker_id, lease())
    tagging.renew(lease().total_seconds())
    IngestQueue().renew(worker_id, lease())
    graphstate.renew(worker_id, lease())
    insights.renew(lease().total_seconds())
    return {**commands, "paused": state["paused"], "shutdown_ok": not state["stay_on"],
            "heartbeat_seconds": settings.worker_heartbeat_seconds}


# --- Controls from the process page ---


def request_stop(mode: str = "after_episode") -> dict:
    """Stop the worker's run after the current episode (abort: cancel it) and pause new work."""
    with _lock:
        update(paused=True)
        _commands.update(stop=True, image_stop=True, abort=mode == "abort")
    busy = bool(last_beat().get("busy"))
    if mode == "abort":
        return {"status": "stopping", "message": "Cancelling the current episode on the worker, then pausing"}
    return {"status": "stopping" if busy else "paused",
            "message": "The worker stops after the current episode and takes no new work until Resume"
            if busy else "Paused: the worker takes no new work until Resume"}


def resume() -> dict:
    with _lock:
        update(paused=False)
        _commands.update(stop=False, abort=False, image_stop=False)
    return {"status": "resumed"}


def stop_images() -> dict:
    if not (last_beat().get("images") or {}).get("is_running"):
        return {"status": "not_running"}
    _commands["image_stop"] = True
    return {"status": "stopping", "message": "The worker finishes the images being read, then links them"}


def cancel_episode(episode_id: str) -> bool:
    """Cancel the episode the worker is on (True), or False if it isn't on it."""
    run = (last_beat().get("run") or {})
    if run.get("episode_id") != episode_id:
        return False
    _commands["cancel_episode"] = episode_id
    return True


def set_stay_on(on: bool) -> dict:
    return update(stay_on=bool(on))


def stop_requested() -> bool:
    return bool(_commands["stop"])


# --- What the process page shows ---


def pending_work() -> dict:
    """Jobs waiting for the worker."""
    from . import insights, tagging
    from .image.queue import ImageQueue
    from .podcast.queue import TranscriptionQueue

    try:
        tags = tagging.pending_count()
    except Exception as e:  # the database mid-rewrite: count it next time
        logger.debug(f"untagged count unavailable: {e}")
        tags = 0
    try:
        cards = insights.pending_count()
    except Exception as e:
        logger.debug(f"insight count unavailable: {e}")
        cards = 0
    from .ingestqueue import IngestQueue

    counts = {"podcasts": TranscriptionQueue().get_pending_count(), "images": ImageQueue().get_pending_count(),
              "ingest": IngestQueue().pending_count(), "tags": tags, "insights": cards}
    counts["total"] = sum(counts.values())
    from . import graphstate

    counts["graph"] = int(graphstate.dirty())  # not in the total: an out-of-date graph alone doesn't wake the PC
    return counts


def run_state() -> dict:
    """The worker's run in runstate.read()'s shape (alive: running and heard from lately)."""
    run = dict(_beat["data"].get("run") or {"phase": "idle"})
    run["alive"] = online() and run.get("phase", "idle") != "idle"
    run["stop_requested"] = stop_requested() or load()["paused"]
    return run


def status() -> dict:
    """The worker for the process page: state offline | waking | idle | working, switches, wake."""
    state = load()
    beat = last_beat()
    wake = state["wake"]
    waking = bool(wake["sent_at"]) and not online()
    return {
        "state": ("working" if beat.get("busy") else "idle") if online() else ("waking" if waking else "offline"),
        "online": online(),
        "worker_id": beat.get("worker_id"),
        "host": beat.get("host"),
        "last_seen_at": state["last_seen_at"],
        "paused": state["paused"],
        "stay_on": state["stay_on"],
        "wake": {**wake, "configured": bool(settings.desktop_mac_address), "window": settings.wake_window},
        "pending": pending_work(),
        "idle_shutdown_minutes": _beat["data"].get("idle_shutdown_minutes", settings.worker_idle_shutdown_minutes),
    }
