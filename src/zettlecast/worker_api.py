"""
The primary's API for the worker (ROLE=primary): /worker/*, authorised by WORKER_TOKEN.

The primary owns the queues and the database; the worker only talks to it here:

    heartbeat       run state etc. in, commands out; renews the leases of what it holds
    claim           a batch of one kind: podcast episodes, images, or notes to tag
    input           download an input the primary holds (an upload, a feed download)
    fetched         the worker has it now: the job refers to the worker's copy ("desktop:..."),
                    the primary's copy is deleted
    started / completed / failed / release   job results; a completed episode or image is
                    stored (embedded, linked) here
    image/add, image/remove                  screenshot triage on the worker queues keeps here
    tag             a note back from the LLM pass
    link            one graph rebuild after a batch
"""

import base64
import hmac
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import paths, workerstate
from .config import settings
from .models import NoteModel

logger = logging.getLogger(__name__)

KINDS = ("podcast", "image", "tag", "insight")  # claimed in this order: one GPU job at a time
HELD_FOLDERS = ("podcasts", "uploads")  # the primary deletes its copy of inputs in these once fetched


def _authorised(request: Request) -> None:
    if not settings.is_primary:
        raise HTTPException(status_code=404, detail="Not the primary (ROLE=primary)")
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not settings.worker_token or not hmac.compare_digest(token, settings.worker_token):
        raise HTTPException(status_code=401, detail="Invalid or missing worker token")


router = APIRouter(prefix="/worker", dependencies=[Depends(_authorised)])


def _held_rel(local: str, default_folder: str) -> str:
    """Where a held input goes on the worker: the same place under its storage folder."""
    rel = paths.relative_to(Path(local), settings.storage_path)
    return rel if rel is not None else f"{default_folder}/{Path(local).name}"


def _delete_held(local: str) -> None:
    """Delete the primary's copy of an input the worker now has (only inputs it downloaded itself)."""
    rel = paths.relative_to(Path(local), settings.storage_path)
    if rel is None or rel.split("/", 1)[0] not in HELD_FOLDERS:
        return
    try:
        Path(local).unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"Could not delete {local} after the worker fetched it: {e}")


# --- Heartbeat and claims ---


@router.post("/heartbeat")
def heartbeat(body: dict, request: Request):
    return workerstate.heartbeat(body, request.client.host if request.client else None)


class ClaimRequest(BaseModel):
    worker_id: str
    kinds: List[str] = list(KINDS)
    limits: dict = {}


@router.post("/claim")
def claim(body: ClaimRequest):
    """The next batch for the worker: {kind, items} (kind None: nothing to do, or paused)."""
    from .image.queue import ImageQueue
    from .podcast.queue import TranscriptionQueue

    if workerstate.load()["paused"]:
        return {"kind": None, "paused": True}
    lease = workerstate.lease()
    for kind in [k for k in KINDS if k in body.kinds]:
        limit = body.limits.get(kind)
        if kind == "podcast":
            items = TranscriptionQueue().claim(body.worker_id, lease, limit)
            if items:
                return {"kind": kind, "items": [{
                    "job_id": i.episode.id,
                    "episode": i.episode.model_dump(),
                    "held": not paths.is_worker_ref(i.episode.audio_path),
                    "rel": None if paths.is_worker_ref(i.episode.audio_path)
                    else _held_rel(i.episode.audio_path, f"podcasts/{i.episode.podcast_name or 'Uploads'}"),
                } for i in items]}
        elif kind == "image":
            items = ImageQueue().claim(body.worker_id, lease, limit)
            if items:
                return {"kind": kind, "items": [{
                    "job_id": i.image.id,
                    "image": i.image.model_dump(),
                    "held": not paths.is_worker_ref(i.image.image_path),
                    "rel": None if paths.is_worker_ref(i.image.image_path)
                    else _held_rel(i.image.image_path, "uploads/images"),
                } for i in items]}
        elif kind == "tag":
            from . import curation, tagging

            notes = tagging.claim(limit or 25, lease.total_seconds())
            if notes:
                return {"kind": kind, "rubric": curation.rubric(),
                        "items": [n.model_dump(mode="json") for n in notes]}
        elif kind == "insight":
            from . import insights

            cards = insights.claim(limit or 20, lease.total_seconds())
            if cards:
                return {"kind": kind, "items": cards}
    return {"kind": None}


# --- Inputs the primary holds ---


def _held_path(kind: str, job_id: str) -> str:
    from .image.queue import ImageQueue
    from .podcast.queue import TranscriptionQueue

    if kind == "podcast":
        item = TranscriptionQueue().items.get(job_id)
        path = item.episode.audio_path if item else None
    elif kind == "image":
        item = ImageQueue().items.get(job_id)
        path = item.image.image_path if item else None
    else:
        raise HTTPException(status_code=404, detail="Unknown kind")
    if not path or paths.is_worker_ref(path):
        raise HTTPException(status_code=404, detail="Not held by the primary")
    return path


@router.get("/input/{kind}/{job_id}")
def input_file(kind: str, job_id: str):
    path = Path(_held_path(kind, job_id))
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Missing on the primary: {path.name}")
    return FileResponse(path, filename=path.name)


class Fetched(BaseModel):
    ref: str  # "desktop:..." where the worker keeps it


@router.post("/{kind}/{job_id}/fetched")
def fetched(kind: str, job_id: str, body: Fetched):
    """The worker has its own copy of the input now: point the job at it, drop the primary's copy."""
    from .image.queue import ImageQueue
    from .podcast.queue import TranscriptionQueue

    if not paths.is_worker_ref(body.ref):
        raise HTTPException(status_code=400, detail="Expected a desktop: reference")
    held = _held_path(kind, job_id)
    queue = TranscriptionQueue() if kind == "podcast" else ImageQueue()
    queue.set_path(job_id, body.ref)
    _delete_held(held)
    return {"status": "ok"}


# --- Podcast results ---


@router.post("/podcast/{job_id}/started")
def podcast_started(job_id: str):
    from .podcast.queue import TranscriptionQueue

    TranscriptionQueue().mark_started(job_id, remote=True)
    return {"status": "ok"}


class EpisodeDone(BaseModel):
    filename: str
    markdown: str
    processing_time_seconds: float = 0.0
    index: Optional[dict] = None  # db.index_payload of the note: chunks + vectors made on the worker


@router.post("/podcast/{job_id}/completed")
def podcast_completed(job_id: str, body: EpisodeDone):
    """Store a transcript note made on the worker: written to transcripts/, indexed (linked by the batch's /worker/link)."""
    from .pipeline import store_transcript
    from .podcast.queue import TranscriptionQueue

    name = Path(body.filename).name
    if not name.endswith(".md"):
        raise HTTPException(status_code=400, detail="Expected a markdown file name")
    path = settings.transcripts_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.markdown, encoding="utf-8")
    note = store_transcript(path, index=body.index, link=False)
    TranscriptionQueue().mark_completed(
        job_id, SimpleNamespace(processing_time_seconds=body.processing_time_seconds), path)
    logger.info(f"Transcript from the worker stored: {note.title}")
    return {"status": "stored", "uuid": note.uuid}


class JobFailed(BaseModel):
    error: str = ""
    cancelled: bool = False


@router.post("/podcast/{job_id}/failed")
def podcast_failed(job_id: str, body: JobFailed):
    from .podcast.queue import TranscriptionQueue

    queue = TranscriptionQueue()
    if body.cancelled:
        queue.mark_cancelled(job_id)
    else:
        queue.mark_failed(job_id, body.error, max_retries=settings.podcast_max_retries)
    return {"status": "ok"}


class JobIds(BaseModel):
    job_ids: List[str]


@router.post("/podcast/release")
def podcast_release(body: JobIds):
    from .podcast.queue import TranscriptionQueue

    return {"released": TranscriptionQueue().release(body.job_ids)}


# --- Image results ---


class ImageDone(BaseModel):
    note: dict
    seconds: float = 0.0
    image_name: Optional[str] = None  # a kept copy of the image (keep with image), base64 below
    image_b64: Optional[str] = None
    index: Optional[dict] = None


@router.post("/image/{job_id}/completed")
def image_completed(job_id: str, body: ImageDone):
    """Store an image note read on the worker (linked by the batch's /worker/link)."""
    from .db import db
    from .image.queue import ImageQueue

    note = NoteModel.model_validate(body.note)
    if body.image_b64 and body.image_name:
        target = settings.storage_path / "images" / Path(body.image_name).name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(body.image_b64))
        note.source_path = str(target)
    db.upsert_note(note, index=body.index)
    ImageQueue().mark_completed(job_id, body.seconds)
    return {"status": "stored", "uuid": note.uuid}


@router.post("/image/{job_id}/failed")
def image_failed(job_id: str, body: JobFailed):
    from .image.queue import ImageQueue

    ImageQueue().mark_failed(job_id, body.error)
    return {"status": "ok"}


@router.post("/image/release")
def image_release(body: JobIds):
    from .image.queue import ImageQueue

    return {"released": ImageQueue().release(body.job_ids)}


class RemoteImage(BaseModel):
    ref: str
    image_hash: str
    title: Optional[str] = None
    collection_name: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    keep_image: bool = True
    focus_region: Optional[dict] = None


@router.post("/image/add")
def image_add(body: RemoteImage):
    """Queue a screenshot kept in triage on the worker."""
    from .image.queue import DuplicateImageError, ImageQueue

    if not paths.is_worker_ref(body.ref):
        raise HTTPException(status_code=400, detail="Expected a desktop: reference")
    try:
        job_id = ImageQueue().add_remote(**body.model_dump())
    except DuplicateImageError as e:
        return {"status": "duplicate", "message": str(e), "job_id": None}
    return {"status": "queued", "job_id": job_id}


class ImageHash(BaseModel):
    image_hash: str


@router.post("/image/remove")
def image_remove(body: ImageHash):
    """Undo of a triage keep: take the image out of the queue unless it was read already."""
    from .image.queue import ImageQueue

    removed = ImageQueue().remove_by_hash(body.image_hash)
    return {"removed": bool(removed), "processed": removed is False}


# --- Tags ---


class Uuids(BaseModel):
    uuids: List[str]


@router.post("/tag/release")
def tag_release(body: Uuids):
    from . import tagging

    tagging.release(body.uuids)
    return {"status": "ok"}


class Tagged(BaseModel):
    note: Optional[dict] = None
    index: Optional[dict] = None
    error: str = ""


@router.post("/tag/{uuid}")
def tag_done(uuid: str, body: Tagged):
    from . import tagging

    if body.note:
        note = NoteModel.model_validate(body.note)
        if note.uuid != uuid:
            raise HTTPException(status_code=400, detail="Note uuid doesn't match")
        tagging.store(note, body.index)
        return {"status": "stored"}
    tagging.failed(uuid, body.error or "failed")
    return {"status": "failed"}


# --- Today's insight cards (insights.py) ---


class Keys(BaseModel):
    keys: List[str]


@router.post("/insight/release")
def insight_release(body: Keys):
    from . import insights

    insights.release(body.keys)
    return {"status": "ok"}


class InsightResult(BaseModel):
    insight: Optional[dict] = None
    error: str = ""


@router.post("/insight/{key}")
def insight_done(key: str, body: InsightResult):
    from . import insights

    insights.store(key, body.insight, None if body.insight else (body.error or "failed"))
    return {"status": "stored" if body.insight else "failed"}


# --- After a batch ---


@router.post("/link")
def link():
    """One graph rebuild for everything the batch stored (edges + communities)."""
    from .linker import rebuild_graph

    return {"edges": rebuild_graph()}
