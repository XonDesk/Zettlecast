"""
Moving the notes from the desktop to the primary (`zettlecast migrate-paths`, run on the primary
once its storage folder has been copied over, with the API stopped).

Paths stored on the desktop are Windows paths under its storage folder (or elsewhere: screenshot
folders). Each one becomes the copy on the primary if that was copied (transcripts, kept images,
uploads), else a reference to the file on the worker ("desktop:podcasts/Show/episode.mp3",
"desktop:C:/Users/me/OneDrive/Screenshots/shot.png"), which the primary fetches from the worker.
Covers the notes in the database, the transcript notes' frontmatter and both queues.
"""

import json
import logging
import re
from pathlib import Path, PureWindowsPath
from typing import Optional

from . import paths, statefile
from .config import settings

logger = logging.getLogger(__name__)

WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
NOTE_CUSTOM_PATHS = ("note_path", "original_path")


def map_path(value, desktop_root: str, local_root: Optional[Path] = None) -> Optional[str]:
    """The new value for a desktop path, or None if it isn't one (URLs, references, local paths)."""
    if not isinstance(value, str) or paths.is_worker_ref(value) or not WINDOWS_PATH.match(value):
        return None
    local_root = Path(local_root or settings.storage_path)
    path = PureWindowsPath(value)
    try:
        rel = path.relative_to(PureWindowsPath(desktop_root))
    except ValueError:
        return paths.PREFIX + path.as_posix()  # outside storage: stays where it is on the desktop
    local = local_root.joinpath(*rel.parts)
    if local.exists():
        return str(local)
    return paths.PREFIX + "/".join(rel.parts)


def migrate_notes(desktop_root: str, dry_run: bool = False) -> int:
    from .db import _q, db

    changed = 0
    rows = db.notes.search().select(["uuid", "source_path", "metadata_json"]).limit(1_000_000).to_list()
    for row in rows:
        values = {}
        new_source = map_path(row["source_path"], desktop_root)
        if new_source:
            values["source_path"] = new_source
        meta = json.loads(row["metadata_json"] or "{}")
        custom = meta.get("custom") or {}
        moved = {k: map_path(custom.get(k), desktop_root) for k in NOTE_CUSTOM_PATHS}
        moved = {k: v for k, v in moved.items() if v}
        if moved:
            custom.update(moved)
            meta["custom"] = custom
            values["metadata_json"] = json.dumps(meta)
        if values:
            changed += 1
            logger.info(f"{row['uuid'][:8]}: {row['source_path']} -> {values.get('source_path', '(same)')}")
            if not dry_run:
                db.notes.update(where=f"uuid = {_q(row['uuid'])}", values=values)
    return changed


def migrate_transcripts(desktop_root: str, dry_run: bool = False) -> int:
    """The frontmatter `source` of transcript notes (a reindex reads it back)."""
    from .identity import parse_frontmatter, serialize_frontmatter

    changed = 0
    for path in sorted(settings.transcripts_path.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        new_source = map_path(meta.get("source"), desktop_root)
        if not new_source:
            continue
        meta["source"] = new_source
        changed += 1
        if not dry_run:
            path.write_text(f"---\n{serialize_frontmatter(meta).strip()}\n---\n{body}", encoding="utf-8")
    return changed


def migrate_queues(desktop_root: str, dry_run: bool = False) -> int:
    changed = 0
    for name, item_key, fields in (("podcast_queue.json", "episode", ("audio_path",)),
                                   ("image_queue.json", "image", ("image_path",))):
        path = settings.storage_path / name
        data = statefile.read(path)
        touched = False
        for item in (data.get("items") or {}).values():
            for field in fields:
                new = map_path((item.get(item_key) or {}).get(field), desktop_root)
                if new:
                    item[item_key][field] = new
                    touched = True
                    changed += 1
            new = map_path(item.get("result_path"), desktop_root)
            if new:
                item["result_path"] = new
                touched = True
        if touched and not dry_run:
            statefile.write(path, data)
    return changed


def migrate(desktop_root: str, dry_run: bool = False) -> dict:
    return {
        "notes": migrate_notes(desktop_root, dry_run),
        "transcripts": migrate_transcripts(desktop_root, dry_run),
        "queue_items": migrate_queues(desktop_root, dry_run),
    }
