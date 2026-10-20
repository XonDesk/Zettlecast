"""
Screenshot triage: folders of screenshots are shown one at a time (web UI: swipe or arrow keys)
and each gets a decision:

- keep: the information goes into the knowledge base (vision note); the image itself isn't kept
- keep_image: the same, plus a copy of the image in storage, shown with the note
- skip: nothing is added

Original files are never changed or deleted. Decisions are keyed by file content, so a screenshot
that is moved or renamed isn't asked about again. State: <storage>/image_triage.json.
"""

import hashlib
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..config import settings

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic"}
DECISIONS = ("keep", "keep_image", "skip")

_lock = threading.Lock()


def state_path() -> Path:
    return settings.storage_path / "image_triage.json"


def _load() -> dict:
    path = state_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {"folders": data.get("folders", []), "decisions": data.get("decisions", {}),
                    "hashes": data.get("hashes", {})}
        except (OSError, ValueError) as e:
            logger.error(f"Triage state unreadable, starting empty: {e}")
    return {"folders": [], "decisions": {}, "hashes": {}}


def _save(state: dict) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(path)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _file_key(path: Path) -> str:
    """Cheap identity for a file version (path, size, mtime) -> its content hash is cached under it."""
    st = path.stat()
    return f"{path}|{st.st_size}|{int(st.st_mtime)}"


def _hash_cached(state: dict, path: Path) -> str:
    key = _file_key(path)
    digest = state["hashes"].get(key)
    if not digest:
        digest = file_hash(path)
        state["hashes"][key] = digest
    return digest


# --- Folders ---


def folders() -> List[str]:
    return _load()["folders"]


def add_folder(folder: str) -> List[str]:
    path = Path(folder).expanduser()
    if not path.is_dir():
        raise ValueError(f"Not a folder: {folder}")
    with _lock:
        state = _load()
        resolved = str(path.resolve())
        if resolved not in state["folders"]:
            state["folders"].append(resolved)
            _save(state)
        return state["folders"]


def remove_folder(folder: str) -> List[str]:
    with _lock:
        state = _load()
        state["folders"] = [f for f in state["folders"] if f != folder]
        _save(state)
        return state["folders"]


def _images_in(folder: str) -> List[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]


def in_folders(path: Path) -> bool:
    """Only files inside a registered folder may be served or decided on."""
    resolved = path.resolve()
    return any(resolved.is_relative_to(Path(f)) for f in folders())


# --- Decisions ---


def pending(limit: int = 20) -> dict:
    """
    The next undecided screenshots, newest first (screenshots taken lately are the ones
    still fresh in mind), plus counts.
    """
    with _lock:
        state = _load()
        files = []
        for folder in state["folders"]:
            files.extend(_images_in(folder))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        decided_keys = {d["key"] for d in state["decisions"].values() if d.get("key")}
        out = []
        undecided = 0
        changed = False
        for path in files:
            key = _file_key(path)
            if key in decided_keys:
                continue
            # content hash only for files not decided by key (renamed or moved copies)
            cached = state["hashes"].get(key)
            if len(out) < limit or cached:
                digest = _hash_cached(state, path)
                changed |= cached is None
                if digest in state["decisions"]:
                    continue
            undecided += 1
            if len(out) < limit:
                st = path.stat()
                out.append({"path": str(path), "name": path.name, "folder": path.parent.name,
                            "size_kb": round(st.st_size / 1024),
                            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")})
        if changed:
            _save(state)
        counts = {d: 0 for d in DECISIONS}
        for d in state["decisions"].values():
            counts[d["decision"]] = counts.get(d["decision"], 0) + 1
        return {"items": out, "remaining": undecided, "decided": counts, "folders": state["folders"]}


def decide(path: Path, decision: str) -> dict:
    """Record a decision; keeps are queued for the vision pass (keep_image copies the image there)."""
    if decision not in DECISIONS:
        raise ValueError(f"Unknown decision: {decision}")
    if not path.is_file() or not in_folders(path):
        raise ValueError("Not a screenshot in a triage folder")
    with _lock:
        state = _load()
        digest = _hash_cached(state, path)
        job_id = None
        if decision != "skip":
            from .queue import DuplicateImageError, ImageQueue

            queue = ImageQueue()
            try:
                job_id = queue.add(path, collection_name="Screenshots")
                queue.set_options(job_id, keep_image=decision == "keep_image")
            except DuplicateImageError:
                logger.info(f"Already in the knowledge base: {path.name}")
        state["decisions"][digest] = {"decision": decision, "path": str(path), "key": _file_key(path),
                                      "job_id": job_id, "at": datetime.now().isoformat(timespec="seconds")}
        _save(state)
        return state["decisions"][digest]


def undo(path: Path) -> Optional[dict]:
    """Forget the decision on a screenshot; a keep still waiting in the queue is taken out again."""
    with _lock:
        state = _load()
        digest = _hash_cached(state, path) if path.is_file() else next(
            (h for h, d in state["decisions"].items() if d["path"] == str(path)), None)
        entry = state["decisions"].pop(digest, None) if digest else None
        if entry and entry.get("job_id"):
            from .queue import ImageQueue

            queue = ImageQueue()
            if queue.remove_pending(entry["job_id"]):
                logger.info(f"Taken out of the image queue: {path.name}")
            else:
                entry["note_kept"] = True  # already processed: the note stays (delete it from the UI)
        _save(state)
        return entry


def decision_counts() -> Dict[str, int]:
    counts = {d: 0 for d in DECISIONS}
    for d in _load()["decisions"].values():
        counts[d["decision"]] = counts.get(d["decision"], 0) + 1
    return counts
