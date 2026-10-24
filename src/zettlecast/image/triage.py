"""
Screenshot triage: folders of screenshots are shown one at a time (web UI: swipe or arrow keys)
and each gets a decision:

- keep: the information goes into the knowledge base (vision note); the image itself isn't kept
- keep_image: the same, plus a copy of the image in storage, shown with the note
- skip: nothing is added, and the screenshot goes to the Recycle Bin RECYCLE_DELAY later (Undo
  until then just cancels it; after that it can be restored from the Recycle Bin)
- ignore: nothing is added, and the screenshot stays where it is

Kept screenshots are never changed. Decisions are keyed by file content, so a screenshot that is
moved or renamed isn't asked about again. State: <storage>/image_triage.json.
"""

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .. import statefile
from ..config import settings

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic"}
DECISIONS = ("keep", "keep_image", "skip", "ignore")
KEEPS = ("keep", "keep_image")  # the decisions that queue the screenshot for the vision pass
RECYCLE_DELAY = timedelta(minutes=10)  # a discarded screenshot waits this long for an Undo

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
    tmp = statefile.tmp_path(path)
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    # on Windows the replace fails while anyone has the file open (a reader in another process);
    # readers are quick, so wait for them rather than lose the decision
    for attempt in range(40):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.025)


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


def recycle(path: Path) -> bool:
    """Send a file to the Windows Recycle Bin (as Explorer's Delete does: it can be restored)."""
    if os.name != "nt":
        raise OSError("The Recycle Bin is only available on Windows")
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 3, 0x4, 0x10, 0x40, 0x400
    op = SHFILEOPSTRUCTW(wFunc=FO_DELETE, pFrom=str(path.resolve()) + "\0",  # list ends with a double null
                         fFlags=FOF_SILENT | FOF_NOCONFIRMATION | FOF_ALLOWUNDO | FOF_NOERRORUI)
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    return result == 0 and not op.fAnyOperationsAborted and not path.exists()


def _sweep(state: dict, now: Optional[datetime] = None) -> bool:
    """Recycle deleted (skip) screenshots whose Undo window has passed. True if anything changed."""
    now = now or datetime.now()
    changed = False
    for digest, entry in state["decisions"].items():
        if entry["decision"] != "skip" or entry.get("recycled") or entry.get("recycle_failed"):
            continue
        if now < datetime.fromisoformat(entry["at"]) + RECYCLE_DELAY:
            continue
        path = Path(entry["path"])
        changed = True
        if not path.is_file():
            entry["recycled"] = "gone"  # moved or deleted meanwhile
            continue
        # only the very file that was discarded: a new file with the same name stays
        if _file_key(path) != entry.get("key") and _hash_cached(state, path) != digest:
            entry["recycled"] = "replaced"
            continue
        try:
            if recycle(path):
                entry["recycled"] = now.isoformat(timespec="seconds")
                logger.info(f"Deleted screenshot sent to the Recycle Bin: {path.name}")
            else:
                entry["recycle_failed"] = True
        except OSError as e:
            logger.warning(f"Could not recycle {path.name}: {e}")
            entry["recycle_failed"] = True
    return changed


def sweep() -> int:
    """Recycle the discarded screenshots that are due (run periodically by the API server)."""
    with _lock:
        state = _load()
        before = sum(1 for d in state["decisions"].values() if d.get("recycled"))
        if _sweep(state):
            _save(state)
        return sum(1 for d in state["decisions"].values() if d.get("recycled")) - before


# --- Folders ---


def folders() -> List[str]:
    with _lock:  # a read during a save would make the save's replace fail
        return _load()["folders"]


def add_folder(folder: str) -> List[str]:
    path = Path(folder.strip()).expanduser()
    # a full path only: an empty or relative one would resolve against the server's own folder
    if not folder.strip() or not path.is_absolute():
        raise ValueError(f"Give the folder's full path: {folder!r}")
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
        changed = _sweep(state)
        stats = {}
        for folder in state["folders"]:
            for path in _images_in(folder):
                try:
                    stats[path] = path.stat()
                except OSError:
                    pass  # gone since the listing (recycled, moved by OneDrive)
        files = sorted(stats, key=lambda p: stats[p].st_mtime, reverse=True)
        decided_keys = {d["key"] for d in state["decisions"].values() if d.get("key")}
        out = []
        undecided = 0
        for path in files:
            key = f"{path}|{stats[path].st_size}|{int(stats[path].st_mtime)}"  # as _file_key, no second stat
            if key in decided_keys:
                continue
            # content hash only for files not decided by key (renamed or moved copies)
            cached = state["hashes"].get(key)
            if len(out) < limit or cached:
                try:
                    digest = _hash_cached(state, path)
                except OSError:
                    continue  # gone meanwhile
                changed |= cached is None
                if digest in state["decisions"]:
                    continue
            undecided += 1
            if len(out) < limit:
                st = stats[path]
                out.append({"path": str(path), "name": path.name, "folder": path.parent.name,
                            "size_kb": round(st.st_size / 1024),
                            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")})
        if changed:
            _save(state)
        counts = {d: 0 for d in DECISIONS}
        for d in state["decisions"].values():
            counts[d["decision"]] = counts.get(d["decision"], 0) + 1
        return {"items": out, "remaining": undecided, "decided": counts, "folders": state["folders"]}


def clean_region(region: Optional[dict]) -> Optional[dict]:
    """A circled region as fractions of the image: box [x, y, w, h] and up to 64 stroke points."""
    if not region:
        return None
    try:
        x, y, w, h = (min(max(float(v), 0.0), 1.0) for v in region["box"])
        points = [[round(min(max(float(px), 0.0), 1.0), 4), round(min(max(float(py), 0.0), 1.0), 4)]
                  for px, py in (region.get("points") or [])]
    except (KeyError, TypeError, ValueError):
        raise ValueError("Bad region")
    if w < 0.01 or h < 0.01:
        return None  # a tap, not a circle
    step = max(1, len(points) // 64)
    return {"box": [round(x, 4), round(y, 4), round(min(w, 1 - x), 4), round(min(h, 1 - y), 4)],
            "points": points[::step]}


def _queue_keep(path: Path, digest: str, keep_image: bool, region: Optional[dict]) -> Optional[str]:
    """
    Queue a kept screenshot for the vision pass: here, or on a worker (ROLE=worker) in the
    primary's queue, as a file on this machine. The job id, or None if it was read already.
    """
    from .queue import DuplicateImageError, ImageQueue

    if settings.is_worker:
        from PIL import Image

        from .. import paths
        from ..worker import primary

        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            width = height = None
        reply = primary().post("/worker/image/add", {
            "ref": paths.to_worker_ref(path), "image_hash": digest, "title": path.stem,
            "collection_name": "Screenshots", "width": width, "height": height,
            "keep_image": keep_image, "focus_region": region,
        })
        if reply.get("status") == "duplicate":
            logger.info(f"Already in the knowledge base: {path.name}")
        return reply.get("job_id")
    queue = ImageQueue()
    try:
        job_id = queue.add(path, collection_name="Screenshots")
        queue.set_options(job_id, keep_image=keep_image, focus_region=region)
        return job_id
    except DuplicateImageError:
        logger.info(f"Already in the knowledge base: {path.name}")
        return None


def _unqueue_keep(job_id: str, digest: Optional[str]) -> bool:
    """Take a keep out of the queue again; False if it was read already."""
    if settings.is_worker:
        from ..worker import primary

        return bool(digest) and bool(primary().post("/worker/image/remove", {"image_hash": digest}).get("removed"))
    from .queue import ImageQueue

    return ImageQueue().remove_pending(job_id)


def decide(path: Path, decision: str, region: Optional[dict] = None) -> dict:
    """
    Record a decision; keeps are queued for the vision pass (keep_image copies the image there).
    region: the part the user circled, read as the screenshot's focus.
    """
    if decision not in DECISIONS:
        raise ValueError(f"Unknown decision: {decision}")
    region = clean_region(region) if decision in KEEPS else None
    if not path.is_file() or not in_folders(path):
        raise ValueError("Not a screenshot in a triage folder")
    with _lock:
        state = _load()
        digest = _hash_cached(state, path)
        job_id = None
        if decision in KEEPS:
            job_id = _queue_keep(path, digest, keep_image=decision == "keep_image", region=region)
        state["decisions"][digest] = {"decision": decision, "path": str(path), "key": _file_key(path),
                                      "job_id": job_id, "at": datetime.now().isoformat(timespec="seconds"),
                                      **({"region": region} if region else {})}
        _save(state)
        return state["decisions"][digest]


def undo(path: Path) -> Optional[dict]:
    """
    Forget the decision on a screenshot: a keep still waiting in the queue is taken out again, a
    discard not yet recycled stays where it is (an already recycled one is in the Recycle Bin).
    """
    with _lock:
        state = _load()
        digest = _hash_cached(state, path) if path.is_file() else next(
            (h for h, d in state["decisions"].items() if d["path"] == str(path)), None)
        entry = state["decisions"].pop(digest, None) if digest else None
        if entry and entry.get("job_id"):
            if _unqueue_keep(entry["job_id"], digest):
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
