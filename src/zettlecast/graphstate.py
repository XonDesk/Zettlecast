"""
Whether the graph needs a rebuild, which the worker does (ROLE=primary).

The primary doesn't compute the graph: storing an indexed note, deleting one or adding a link
edge marks it out of date (mark_dirty), and the worker claims one "graph" job when it has
nothing else left to do, so a session ends with one rebuild instead of one per batch. A version
counter keeps changes made while the worker computes: the graph is only up to date once a
rebuild started at the latest version is stored.

<storage>/graph_state.json: {"version", "built_version", "job": {id, worker_id, version, lease_until}}
"""

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from . import statefile
from .config import settings

logger = logging.getLogger(__name__)


def _path() -> Path:
    return settings.storage_path / "graph_state.json"


def _load() -> dict:
    data = statefile.read(_path())
    data.setdefault("version", 0)
    data.setdefault("built_version", 0)
    data.setdefault("job", None)
    return data


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def mark_dirty() -> None:
    with statefile.lock:
        data = _load()
        data["version"] += 1
        statefile.write(_path(), data)


def dirty() -> bool:
    data = _load()
    return data["version"] > data["built_version"]


def claim(worker_id: str, lease: timedelta) -> Optional[dict]:
    """The rebuild job, if the graph is out of date and nobody holds it."""
    with statefile.lock:
        data = _load()
        job = data["job"]
        if data["version"] <= data["built_version"] or (job and job["lease_until"] > _now()):
            return None
        data["job"] = {"id": uuid.uuid4().hex[:12], "worker_id": worker_id, "version": data["version"],
                       "lease_until": (datetime.utcnow() + lease).isoformat(timespec="seconds")}
        statefile.write(_path(), data)
        return dict(data["job"])


def renew(worker_id: str, lease: timedelta) -> None:
    with statefile.lock:
        data = _load()
        if data["job"] and data["job"]["worker_id"] == worker_id:
            data["job"]["lease_until"] = (datetime.utcnow() + lease).isoformat(timespec="seconds")
            statefile.write(_path(), data)


def finish(job_id: str, stored: bool) -> bool:
    """The job is done (stored: its graph is in the database) or handed back. False if it wasn't the current job."""
    with statefile.lock:
        data = _load()
        if not data["job"] or data["job"]["id"] != job_id:
            return False
        if stored:
            data["built_version"] = max(data["built_version"], data["job"]["version"])
        data["job"] = None
        statefile.write(_path(), data)
        return True
