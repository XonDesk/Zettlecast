"""
JSON state files that more than one writer updates (the podcast and image queues).

Each writer loads the whole file, changes a few items and saves it again. save() merges three
ways - what this writer loaded (base), what it holds now (mine) and what is on disk now
(theirs) - so items this writer changed win and every other item keeps what is on disk. Two
writers that each loaded the file never undo each other's changes: a feed import downloading
for minutes no longer wipes out the jobs the worker finished meanwhile.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

lock = threading.RLock()  # one read-merge-write at a time in this process


def read(path: Path) -> dict:
    """The file's data ({} if missing or unreadable)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.error(f"Could not read {path.name}: {e}")
        return {}


def tmp_path(path: Path) -> Path:
    """
    A temporary file of this writer's own, to write and then rename over path: two processes
    saving at once (the API server and a CLI run) would otherwise rename each other's away.
    """
    return path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")


def write(path: Path, data: dict) -> None:
    """Write via a temporary file: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(path)
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    for attempt in range(10):  # a reader may hold the file open for a moment on Windows
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)


def merge(base: dict, mine: dict, theirs: dict, item_keys=("items",), list_keys: dict = None) -> dict:
    """
    item_keys: dicts of items (job id -> item); an item added, changed or removed here wins.
    list_keys: append-only lists (processing times, completed hashes) -> how many entries to keep;
    entries added here are appended to what is on disk. Other keys: this writer's value if it
    changed it.
    """
    list_keys = list_keys or {}
    out = dict(theirs)
    for key in set(mine) - set(item_keys) - set(list_keys):
        if mine[key] != base.get(key) or key not in theirs:
            out[key] = mine[key]
    for key in item_keys:
        b, m, t = base.get(key) or {}, mine.get(key) or {}, theirs.get(key) or {}
        merged = dict(t)
        for k in list(b) + [k for k in m if k not in b]:
            if k in m and m[k] != b.get(k):
                merged[k] = m[k]
            elif k not in m and k in b:
                merged.pop(k, None)
        out[key] = merged
    for key, keep in list_keys.items():
        b, m, t = base.get(key) or [], mine.get(key) or [], theirs.get(key) or []
        added = m[len(b):] if m[:len(b)] == b else [x for x in m if x not in b]
        out[key] = (t + added)[-keep:]
    return out


def save(path: Path, base: dict, mine: dict, item_keys=("items",), list_keys: dict = None) -> dict:
    """Merge with the file on disk, write it, and return what was written (the new base)."""
    with lock:
        merged = merge(base, mine, read(path), item_keys=item_keys, list_keys=list_keys)
        write(path, merged)
        return merged
