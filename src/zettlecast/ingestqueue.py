"""
Pages and files added on the primary, for the worker to read (ROLE=primary).

The primary doesn't parse: a web address, an uploaded PDF or markdown file, or a link fetched
from a note is queued here, with a placeholder note at once (listed with "waiting for the
desktop", not searchable or linked). The worker claims a batch under a lease, fetches the page or
downloads the file (the primary keeps its copy: a PDF stays readable with the desktop off),
parses it, runs the LLM and embeds it, and sends the note back under the placeholder's uuid;
complete() stores it in the placeholder's place. After MAX_ATTEMPTS failures the placeholder
says why (custom["ingest_error"]).

<storage>/ingest_queue.json, saved with statefile's three-way merge like the other queues.
"""

import json
import logging
import uuid as uuidlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from . import statefile
from .config import settings
from .models import NoteModel

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
PLACEHOLDER = "ingest_pending"  # metadata.custom: the job a placeholder note waits for
ERROR = "ingest_error"  # metadata.custom: why it couldn't be added, once given up
FILE_TYPES = {".pdf": "pdf", ".md": "markdown", ".markdown": "markdown", ".txt": "text"}


def _path() -> Path:
    return settings.storage_path / "ingest_queue.json"


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


class IngestQueue:
    def __init__(self):
        self._base = statefile.read(_path())
        self.items: dict = json.loads(json.dumps(self._base.get("items", {})))

    def save(self) -> None:
        merged = statefile.save(_path(), self._base, {"items": self.items})
        self._base = merged
        self.items = json.loads(json.dumps(merged.get("items", {})))

    # --- On the primary ---

    def add_url(self, url: str, referenced_by: Optional[NoteModel] = None) -> dict:
        parsed = urlparse(url)
        return self._add("url", title=f"{parsed.netloc}{parsed.path}".rstrip("/") or url, url=url,
                         source_type="web", source_path=url, referenced_by=referenced_by)

    def add_file(self, path: Path) -> dict:
        path = Path(path)
        return self._add("file", title=path.stem, path=str(path), source_path=str(path),
                         source_type=FILE_TYPES.get(path.suffix.lower(), "markdown"))

    def _add(self, kind: str, title: str, source_type: str, source_path: str, url: str = None,
             path: str = None, referenced_by: Optional[NoteModel] = None) -> dict:
        from .db import db

        job_id = uuidlib.uuid4().hex[:12]
        placeholder = NoteModel(title=title, source_type=source_type, source_path=source_path,
                                full_text=url or Path(path).name, content_hash=f"queued:{job_id}")
        placeholder.metadata.custom[PLACEHOLDER] = job_id
        db.store_placeholder(placeholder)
        self.items[job_id] = {
            "id": job_id, "kind": kind, "url": url, "path": path, "title": title,
            "placeholder": placeholder.uuid,
            "referenced_by": referenced_by.uuid if referenced_by else None,
            "referenced_by_title": referenced_by.title if referenced_by else None,
            "status": "pending", "attempts": 0, "error": None, "worker_id": None, "lease_until": None,
            "added_at": _now(), "finished_at": None, "seconds": None, "note": None,
        }
        self.save()
        logger.info(f"Queued for the desktop: {url or path}")
        return self.items[job_id]

    def pending_count(self) -> int:
        self._expire()
        return sum(i["status"] in ("pending", "processing") for i in self.items.values())

    def _expire(self) -> None:
        """A claim whose lease ran out (the worker stopped) goes back to pending."""
        now = _now()
        expired = [i for i in self.items.values()
                   if i["status"] == "processing" and i["lease_until"] and i["lease_until"] < now]
        for item in expired:
            item.update(status="pending", worker_id=None, lease_until=None)
        if expired:
            self.save()

    # --- For the worker (worker_api.py) ---

    def claim(self, worker_id: str, lease: timedelta, limit: Optional[int] = None) -> List[dict]:
        self._expire()
        pending = [i for i in self.items.values() if i["status"] == "pending"][: limit or None]
        until = (datetime.utcnow() + lease).isoformat(timespec="seconds")
        for item in pending:
            item.update(status="processing", worker_id=worker_id, lease_until=until,
                        attempts=item["attempts"] + 1)
        if pending:
            self.save()
        return pending

    def renew(self, worker_id: str, lease: timedelta) -> int:
        held = [i for i in self.items.values() if i["status"] == "processing" and i["worker_id"] == worker_id]
        until = (datetime.utcnow() + lease).isoformat(timespec="seconds")
        for item in held:
            item["lease_until"] = until
        if held:
            self.save()
        return len(held)

    def release(self, job_ids: List[str]) -> int:
        """Claimed but not started (a stop): back to pending, the attempt not counted."""
        released = 0
        for job_id in job_ids:
            item = self.items.get(job_id)
            if item and item["status"] == "processing":
                item.update(status="pending", worker_id=None, lease_until=None, attempts=max(0, item["attempts"] - 1))
                released += 1
        if released:
            self.save()
        return released

    def complete(self, job_id: str, note: NoteModel, index: Optional[dict], seconds: float = 0.0) -> dict:
        """Store the worker's note in its placeholder's place (or record a duplicate)."""
        from .db import db

        item = self.items.get(job_id)
        if not item:
            raise KeyError(job_id)
        note.uuid = item["placeholder"]  # the worker sends it so already; the chunk ids depend on it
        if item["kind"] == "file":
            note.source_path = item["path"]  # the primary's copy, readable with the desktop off
        existing = db.get_note_by_hash(note.content_hash)
        if existing and existing.uuid != note.uuid:
            db.delete_note(note.uuid)
            if item["referenced_by"]:
                from .references import add_referrer

                source = db.get_note_by_uuid(item["referenced_by"])
                if source:
                    add_referrer(existing, source)
            item.update(status="duplicate", note=existing.uuid, finished_at=_now(), seconds=seconds,
                        worker_id=None, lease_until=None)
        else:
            db.upsert_note(note, index=index)
            item.update(status="completed", note=note.uuid, title=note.title, finished_at=_now(),
                        seconds=seconds, worker_id=None, lease_until=None)
        self.save()
        return item

    def fail(self, job_id: str, error: str) -> dict:
        from .db import db

        item = self.items.get(job_id)
        if not item:
            raise KeyError(job_id)
        item.update(error=(error or "failed")[:500], worker_id=None, lease_until=None)
        if item["attempts"] < MAX_ATTEMPTS:
            item["status"] = "pending"
        else:
            item.update(status="failed", finished_at=_now())
            placeholder = db.get_note_by_uuid(item["placeholder"])
            if placeholder:
                custom = placeholder.metadata.custom
                custom.pop(PLACEHOLDER, None)
                custom[ERROR] = item["error"]
                db.update_note_metadata(placeholder.uuid, placeholder.metadata.model_dump())
        self.save()
        return item
