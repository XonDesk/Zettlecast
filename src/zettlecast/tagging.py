"""
LLM tags + summaries and indexes for notes stored without them, when the machine storing notes
has no LLM and doesn't embed (the primary).

On the primary (ROLE=primary) web pages, PDFs, markdown and Reddit saves are stored at once,
without the LLM and without an index (db.INDEX_PENDING). What still needs work is read from the
database (no summary yet; a Reddit save with its thread fetched but not curated; waiting for an
index), so nothing has to be queued at ingest. The worker claims a batch, runs the LLM pass a
standalone install runs at ingest where it's needed, embeds the note and sends it back with its
index; the primary stores it as sent and links the batch with one graph rebuild.

A note the LLM fails on is tried MAX_ATTEMPTS times (<storage>/tag_failures.json), then left.
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .config import settings
from .enricher import MIN_TEXT  # enrich_note leaves shorter notes alone
from .models import NoteModel

logger = logging.getLogger(__name__)

TAGGED_TYPES = ("web", "pdf", "markdown", "text", "reddit")  # what parse_file / parse_url / reddit enrich
MAX_ATTEMPTS = 3
CACHE_SECONDS = 60

_lock = threading.Lock()
_claimed: dict = {}  # uuid -> claim expiry (monotonic): handed to the worker, not handed out again
_cache: dict = {"at": 0.0, "uuids": None}


def _failures_path() -> Path:
    return settings.storage_path / "tag_failures.json"


def _failures() -> dict:
    try:
        return json.loads(_failures_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_failures(data: dict) -> None:
    from . import statefile

    statefile.write(_failures_path(), data)


def needs_llm(source_type: str, metadata: dict, text: str) -> bool:
    """Whether a stored note still waits for the LLM pass it would have had at ingest."""
    if source_type not in TAGGED_TYPES:
        return False
    if source_type == "reddit":
        from .reddit import needs_llm as reddit_needs_llm

        return reddit_needs_llm(metadata)
    return "summary" not in (metadata.get("custom") or {}) and len(text.strip()) >= MIN_TEXT


def _untagged() -> List[str]:
    """
    Uuids of stored notes that need the LLM pass or an index, newest first (cached for a minute:
    the wake policy asks often).
    """
    if _cache["uuids"] is not None and time.monotonic() - _cache["at"] < CACHE_SECONDS:
        return _cache["uuids"]
    from .db import INDEX_PENDING, _q, db

    types = ", ".join(_q(t) for t in TAGGED_TYPES)
    rows = (db.notes.search().where(f"source_type IN ({types})")
            .select(["uuid", "source_type", "metadata_json", "full_text", "created_at"])
            .limit(1_000_000).to_list())
    # waiting for an index, any type (a transcript's text isn't read for this: uuids only)
    rows += (db.notes.search().where(f"metadata_json LIKE '%\"{INDEX_PENDING}\": true%'")
             .select(["uuid", "created_at"]).limit(1_000_000).to_list())
    rows.sort(key=lambda r: r["created_at"], reverse=True)  # newest first: what was just added
    uuids = list(dict.fromkeys(
        r["uuid"] for r in rows
        if "metadata_json" not in r  # from the waiting query
        or needs_index(json.loads(r["metadata_json"] or "{}"))
        or needs_llm(r["source_type"], json.loads(r["metadata_json"] or "{}"), r["full_text"] or "")))
    _cache.update(at=time.monotonic(), uuids=uuids)
    return uuids


def needs_index(metadata: dict) -> bool:
    from .db import INDEX_PENDING

    return bool((metadata.get("custom") or {}).get(INDEX_PENDING))


def forget_cache() -> None:
    _cache.update(at=0.0, uuids=None)


def pending_count() -> int:
    if not settings.enable_auto_tagging:
        return 0
    failures = _failures()
    return sum(1 for u in _untagged() if failures.get(u, {}).get("attempts", 0) < MAX_ATTEMPTS)


def claim(limit: int, lease_seconds: float) -> List[NoteModel]:
    """Notes for the worker's LLM pass, held for lease_seconds (a heartbeat renews them)."""
    if not settings.enable_auto_tagging:
        return []
    from .db import db

    failures = _failures()
    now = time.monotonic()
    notes = []
    with _lock:
        for uuid, until in list(_claimed.items()):
            if until < now:
                del _claimed[uuid]
        for uuid in _untagged():
            if len(notes) >= limit:
                break
            if uuid in _claimed or failures.get(uuid, {}).get("attempts", 0) >= MAX_ATTEMPTS:
                continue
            note = db.get_note_by_uuid(uuid)
            if note is not None:
                notes.append(note)
                _claimed[uuid] = now + lease_seconds
    return notes


def renew(lease_seconds: float) -> None:
    with _lock:
        until = time.monotonic() + lease_seconds
        for uuid in _claimed:
            _claimed[uuid] = until


def release(uuids: List[str]) -> None:
    with _lock:
        for uuid in uuids:
            _claimed.pop(uuid, None)


def store(note: NoteModel, index: Optional[dict] = None) -> None:
    """A note back from the worker's LLM pass: stored with the vectors it sent (linked by the batch's rebuild)."""
    from .db import db

    if db.get_note_by_uuid(note.uuid) is None:
        logger.info(f"Tagged note was deleted meanwhile: {note.title}")
    elif not db.upsert_note(note, index=index):
        # stored, but waiting again: the index didn't fit (or none came). Counted, so it isn't
        # handed out forever
        failed(note.uuid, "the worker's index didn't fit the note")
        return
    else:
        if note.source_type == "reddit":
            from .reddit import mark_tagged

            mark_tagged(note)
    with _lock:
        _claimed.pop(note.uuid, None)
        failures = _failures()
        if failures.pop(note.uuid, None) is not None:
            _save_failures(failures)
        if _cache["uuids"] is not None and note.uuid in _cache["uuids"]:
            _cache["uuids"] = [u for u in _cache["uuids"] if u != note.uuid]


def failed(uuid: str, error: str) -> None:
    with _lock:
        _claimed.pop(uuid, None)
        failures = _failures()
        entry = failures.setdefault(uuid, {"attempts": 0})
        entry.update(attempts=entry["attempts"] + 1, error=error[:300],
                     at=datetime.now().isoformat(timespec="seconds"))
        _save_failures(failures)


# --- On the worker ---


def llm_pass(note: NoteModel) -> Tuple[Optional[NoteModel], str]:
    """
    The LLM work a note gets at ingest on a standalone install. Returns (note, "") or
    (None, why) when the LLM couldn't do it.
    """
    if note.source_type == "reddit":
        from .reddit import llm_pass as reddit_pass

        done, _ = reddit_pass(note)
        return (done, "") if done is not None else (None, "LLM unavailable or no usable answer")
    from .enricher import enrich_note

    note = enrich_note(note)
    if "summary" in note.metadata.custom or len(note.full_text.strip()) < MIN_TEXT:
        return note, ""
    return None, "LLM unavailable or no usable answer"
