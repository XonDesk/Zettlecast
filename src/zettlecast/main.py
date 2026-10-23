"""
Zettlecast FastAPI Main Application
API endpoints for ingestion, search, and note management.
"""

import asyncio
import json
import logging
import mimetypes
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from . import embeddings, llm, paths, workerstate
from .config import ENV_FILE, settings
from .brief import has_image, iso_date, note_brief
from .db import _q, db
from .models import ProcessingResult
from .parser import AUDIO_EXTENSIONS, parse_url
from .pipeline import ingest_note, ingest_path
from .search import accept_link, get_suggestions_for_note, reject_link, search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
for _noisy in ("nemo_logger", "nemo", "lightning", "pytorch_lightning", "httpx", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _start_recycle_sweeper() -> None:
    """Discarded screenshots go to the Recycle Bin once their Undo window passes, page open or not."""
    def loop() -> None:
        from .image import triage

        while True:
            try:
                triage.sweep()
            except Exception as e:
                logger.warning(f"Screenshot recycle sweep failed: {e}")
            time.sleep(60)

    threading.Thread(target=loop, name="screenshot-recycler", daemon=True).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting Zettlecast API...")
    settings.ensure_directories()
    db.connect()
    logger.info(f"Storage path: {settings.storage_path}")
    # free the GPU for transcription runs when nobody is searching
    embeddings.start_idle_unloader(settings.model_idle_minutes)
    if settings.is_primary:
        # GPU work and the screenshots are on the worker: wake it when jobs wait for it
        from .wake import start_policy

        start_policy()
        logger.info(f"Primary: jobs go to the worker (wake-on-LAN {'on' if settings.desktop_mac_address else 'off'})")
    else:
        _start_recycle_sweeper()
    yield
    llm.unload(only_own=True)


app = FastAPI(
    title="Zettlecast",
    description="Digital Zettelkasten API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend and bookmarklet
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bookmarklet needs this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth Middleware ---

@app.middleware("http")
async def verify_token(request: Request, call_next):
    """Verify API token for mutation endpoints."""
    # Allow OPTIONS for CORS preflight
    if request.method == "OPTIONS":
        return await call_next(request)
    
    # Public endpoints
    public_paths = ["/", "/docs", "/openapi.json", "/health"]
    if request.url.path in public_paths:
        return await call_next(request)

    # The worker's API checks the worker token itself (worker_api.py)
    if request.url.path.startswith("/worker/"):
        return await call_next(request)
    
    # Check token
    token = request.query_params.get("token") or request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != settings.api_token:
        return JSONResponse(status_code=401, content={"error": "Invalid or missing API token"})
    
    return await call_next(request)


from .worker_api import router as _worker_router  # noqa: E402

app.include_router(_worker_router)


# --- Health Check ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0", "role": settings.role}


# --- The worker's files, through the primary (ROLE=primary) ---

DESKTOP_OFFLINE = "The desktop is off: wake it from the process page (or wait for it to finish starting)"


def _worker_base() -> str:
    url = workerstate.worker_url()
    if not url or not workerstate.online():
        raise HTTPException(status_code=503, detail=DESKTOP_OFFLINE)
    return url


def _worker_headers() -> dict:
    return {"Authorization": f"Bearer {settings.worker_token}"}


async def _worker_json(method: str, path: str, params: Optional[dict] = None, body: Optional[dict] = None):
    """Call the worker's file server and return its JSON (its errors pass through)."""
    base = _worker_base()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.request(method, base + path, params=params, json=body, headers=_worker_headers())
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail=DESKTOP_OFFLINE)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail") or r.text
        except ValueError:
            detail = r.text
        raise HTTPException(status_code=r.status_code, detail=detail)
    return r.json()


PASSED_HEADERS = ("content-type", "content-length", "content-range", "accept-ranges", "content-disposition",
                  "cache-control", "etag", "last-modified")


async def _worker_file(path: str, params: dict, request: Request) -> StreamingResponse:
    """Stream a file from the worker, Range requests included (seeking in audio)."""
    base = _worker_base()
    headers = _worker_headers()
    if request.headers.get("range"):
        headers["Range"] = request.headers["range"]
    client = httpx.AsyncClient(timeout=httpx.Timeout(60, read=None))
    try:
        upstream = await client.send(client.build_request("GET", base + path, params=params, headers=headers),
                                     stream=True)
    except httpx.HTTPError:
        await client.aclose()
        raise HTTPException(status_code=503, detail=DESKTOP_OFFLINE)
    if upstream.status_code >= 400:
        await upstream.aclose()
        await client.aclose()
        raise HTTPException(status_code=upstream.status_code, detail="Not found on the desktop")

    async def close() -> None:
        await upstream.aclose()
        await client.aclose()

    return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code,
                             headers={k: v for k, v in upstream.headers.items() if k.lower() in PASSED_HEADERS},
                             background=BackgroundTask(close))


def _work_for_worker(what: str) -> dict:
    """A run asked for on the primary: the worker picks the jobs up (unpaused, woken if off)."""
    from .wake import wake_now

    workerstate.resume()
    if workerstate.online():
        return {"status": "started", "message": f"The desktop picks up the {what} next"}
    woken = wake_now(reason=f"{what} asked for")
    if woken["status"] == "sent":
        return {"status": "waking", "message": f"Waking the desktop: it starts on the {what} once it is up"}
    return {"status": "queued", "message": f"Queued: the desktop does the {what} when it is next on "
                                           f"({woken['message']})"}


# --- Ingestion Endpoints ---

class IngestResponse(BaseModel):
    status: str
    uuid: Optional[str] = None
    title: Optional[str] = None
    error: Optional[str] = None


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    token: str = Query(...),
    url: Optional[str] = Query(None, description="URL to ingest"),
    file: Optional[UploadFile] = File(None, description="File to ingest"),
):
    """
    Ingest content from URL or file upload.
    
    Provide either `url` OR `file`, not both.
    """
    if url and file:
        raise HTTPException(status_code=400, detail="Provide either url or file, not both")
    
    if not url and not file:
        raise HTTPException(status_code=400, detail="Provide either url or file")
    
    result: ProcessingResult

    if url:
        existing = db.get_note_by_source_path(url)
        if existing:
            return IngestResponse(
                status="duplicate",
                uuid=existing.uuid,
                title=existing.title,
                error=f"URL already exists as '{existing.title}'",
            )
        logger.info(f"Ingesting URL: {url}")
        result = parse_url(url)
        if result.status == "success" and result.note:
            ingest_note(result.note)
    else:
        # Keep uploads so the note's source stays viewable; audio goes to the podcast folder
        name = re.sub(r"[^\w.\- ]", "_", Path(file.filename or "upload").name)
        target_dir = (
            settings.podcasts_path / "Uploads"
            if Path(name).suffix.lower() in AUDIO_EXTENSIONS
            else settings.storage_path / "uploads"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / name
        if target.exists():
            target = target_dir / f"{target.stem}_{datetime.now():%Y%m%d%H%M%S}{target.suffix}"
        target.write_bytes(await file.read())
        logger.info(f"Ingesting file: {target}")
        result = ingest_path(target, podcast_name="Uploads")

    if result.status == "failed":
        return IngestResponse(status="failed", error=result.error_message)

    return IngestResponse(status=result.status, uuid=result.uuid, title=result.title)


# --- Note Endpoints ---

@app.get("/notes")
async def list_notes(
    token: str = Query(...),
    status: Optional[str] = Query(None, description="Filter by status"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    limit: int = Query(50, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    """List all notes with optional filtering."""
    notes = db.list_notes(status=status, source_type=source_type, limit=limit, offset=offset)
    return {"notes": notes, "count": len(notes)}


@app.get("/notes/{uuid}")
async def get_note(
    uuid: str,
    token: str = Query(...),
    include_suggestions: bool = Query(True),
):
    """Get a single note by UUID with optional suggestions."""
    note = db.get_note_by_uuid(uuid)
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    response = {
        "uuid": note.uuid,
        "title": note.title,
        "source_type": note.source_type,
        "source_path": note.source_path,
        "full_text": note.full_text,
        "status": note.status,
        "created_at": note.created_at.isoformat(),
        "metadata": note.metadata.model_dump(),
    }
    
    if include_suggestions:
        suggestions = get_suggestions_for_note(uuid, limit=50)  # all of its links, strongest first
        response["suggestions"] = [s.model_dump() for s in suggestions]
    
    return response


@app.delete("/notes/{uuid}")
async def delete_note(
    uuid: str,
    token: str = Query(...),
):
    """Delete a note by UUID."""
    if not db.get_note_by_uuid(uuid):
        raise HTTPException(status_code=404, detail="Note not found")

    db.delete_note(uuid)
    return {"status": "deleted", "uuid": uuid}


@app.get("/notes/{uuid}/source")
async def get_note_source_file(
    uuid: str,
    request: Request,
    token: str = Query(...),
):
    """Download/view the original source file for a note."""
    note = db.get_note_by_uuid(uuid)

    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    source_path = Path(note.source_path)
    if note.metadata.custom.get("image_kept") is False:
        # screenshot kept for its information only
        raise HTTPException(status_code=404, detail="Image not kept for this note")

    # Check if it's a URL
    if note.source_path.startswith(('http://', 'https://')):
        return {"url": note.source_path, "type": "url"}

    # On the worker (podcast audio, screenshots): streamed from there while it is on
    if paths.is_worker_ref(note.source_path):
        return await _worker_file("/files", {"ref": note.source_path}, request)

    # Check if file exists
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source file not found")

    media_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
    return FileResponse(
        source_path,
        media_type=media_type,
        filename=source_path.name,
    )


THUMB_MAX = 720  # px, longest side: sharp at the graph's ~240 px preview on a 2x screen, ~50 KB


@app.get("/notes/{uuid}/thumb")
def get_note_thumbnail(uuid: str, token: str = Query(...)):
    """
    A small JPEG of a kept screenshot, for the graph's hover preview (originals are ~2 MB). Made
    once into <storage>/thumbs; an image note's uuid comes from its bytes, so it never goes stale.
    """
    from PIL import Image, ImageOps

    note = db.get_note_by_uuid(uuid)
    if not note or not has_image(note.source_type, note.metadata.custom):
        raise HTTPException(status_code=404, detail="No image for this note")
    thumb = settings.storage_path / "thumbs" / f"{uuid}.jpg"
    if not thumb.exists():
        source = Path(note.source_path)
        if paths.is_worker_ref(note.source_path) or not source.exists():
            raise HTTPException(status_code=404, detail="Image file not here")
        thumb.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((THUMB_MAX, THUMB_MAX))
            image.save(thumb, "JPEG", quality=80)
    return FileResponse(thumb, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=604800"})


@app.get("/notes/{uuid}/markdown")
async def get_note_markdown(
    uuid: str,
    token: str = Query(...),
):
    """Download note as markdown file."""
    note = db.get_note_by_uuid(uuid)

    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Generate markdown content
    markdown_lines = [
        f"# {note.title}",
        "",
        f"**UUID**: {note.uuid}",
        f"**Source**: {note.source_type}",
        f"**Created**: {note.created_at.isoformat()}",
        "",
    ]

    # Add metadata
    if note.metadata.tags:
        markdown_lines.append(f"**Tags**: {', '.join(note.metadata.tags)}")
        markdown_lines.append("")

    # Add source link
    if note.source_path:
        if note.source_path.startswith(('http://', 'https://')):
            markdown_lines.append(f"**Source URL**: [{note.source_path}]({note.source_path})")
        else:
            markdown_lines.append(f"**Source File**: `{note.source_path}`")
        markdown_lines.append("")

    # Add full text
    markdown_lines.append("---")
    markdown_lines.append("")
    markdown_lines.append(note.full_text)

    markdown_content = "\n".join(markdown_lines)

    # Return as downloadable file
    return Response(
        content=markdown_content.encode('utf-8'),
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="{note.uuid[:8]}_{note.title[:30]}.md"'
        },
    )


# --- Search Endpoints ---

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    rerank: bool = True


@app.get("/search")
async def search_notes(
    q: str = Query(..., description="Search query"),
    token: str = Query(...),
    top_k: int = Query(5, ge=1, le=50),
    rerank: bool = Query(True),
):
    """Search notes using semantic similarity."""
    if len(q) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters")
    
    results = search(query=q, top_k=top_k, rerank_results=rerank)
    
    return {
        "query": q,
        "results": [r.model_dump() for r in results],
        "count": len(results),
    }


# --- Link Management ---

class LinkAction(BaseModel):
    target_uuid: str
    action: str  # "accept" or "reject"


@app.post("/notes/{uuid}/link")
async def manage_link(
    uuid: str,
    body: LinkAction,
    token: str = Query(...),
):
    """Accept or reject a link suggestion."""
    if body.action not in ["accept", "reject"]:
        raise HTTPException(status_code=400, detail="Action must be 'accept' or 'reject'")
    
    if body.action == "accept":
        success = accept_link(uuid, body.target_uuid)
        return {"status": "linked" if success else "already_linked"}
    else:
        reject_link(uuid, body.target_uuid)
        return {"status": "rejected"}


# --- Today's insights (insights.py) ---


@app.get("/today")
def get_today(token: str = Query(...)):
    """Today's cards: graph connections worth a look, with the LLM's take once it has one."""
    from . import insights

    result = insights.today()
    if result["preparing"] and insights.generate_in_background():  # standalone; a primary's worker claims them
        result["generating"] = True
    return result


class InsightVerdict(BaseModel):
    verdict: str  # good | rejected


@app.post("/today/{key}/verdict")
def decide_insight(key: str, body: InsightVerdict, token: str = Query(...)):
    """good: worth the card; rejected: not related (the link leaves the graph for good)."""
    from . import insights

    if body.verdict not in ("good", "rejected"):
        raise HTTPException(status_code=400, detail="verdict must be good or rejected")
    try:
        insights.decide(key, body.verdict)
    except KeyError:
        raise HTTPException(status_code=404, detail="No such card")
    return {"status": body.verdict}


class InsightNote(BaseModel):
    title: str
    text: str


@app.post("/today/{key}/note")
def note_from_insight(key: str, body: InsightNote, token: str = Query(...)):
    """A note of your own from a card, [[linked]] to both notes (in <storage>/notes)."""
    from . import insights

    try:
        uuid = insights.write_note(key, body.title, body.text)
    except KeyError:
        raise HTTPException(status_code=404, detail="No such card")
    return {"status": "noted", "uuid": uuid}


# --- Graph Data ---

def _graph_facets(note: dict, row: dict) -> dict:
    """What the graph view filters and colours by: topics, people, organisations, source, date."""
    from .linker import source_group

    meta = json.loads(row.get("metadata_json") or "{}")
    custom = meta.get("custom") or {}
    podcast = custom.get("podcast") or {}
    group = source_group({"source_type": note["source_type"], "source_path": row.get("source_path"),
                          "metadata": meta})
    return {
        "tags": [str(t) for t in meta.get("tags") or []],
        "people": [str(p) for p in custom.get("people") or [] if p],
        "organisations": [str(o) for o in custom.get("organisations") or [] if o],
        "source": group.split(":", 1)[-1],  # show name, website host, subreddit, or type
        "date": iso_date(podcast.get("published")) or iso_date((custom.get("reddit") or {}).get("created"))
                or str(note.get("created_at") or "")[:10] or None,
        "duration": meta.get("duration_seconds"),
    }


@app.get("/graph")
async def get_graph_data(
    token: str = Query(...),
    limit: int = Query(2000, ge=1, le=10000),
):
    """
    Graph nodes (notes), links (edges) and communities for the graph view.

    A note added since the last rebuild has no community yet: it takes its neighbours' most
    common one (weighted), or none.
    """
    notes = db.list_notes(limit=limit)
    ids = {n["uuid"] for n in notes}
    details = {
        r["uuid"]: r
        for r in db.notes.search().select(["uuid", "source_path", "metadata_json"]).limit(100_000).to_list()
        if r["uuid"] in ids
    }
    edges = [e for e in db.get_all_edges() if e["source_uuid"] in ids and e["target_uuid"] in ids]
    links = [
        {"source": e["source_uuid"], "target": e["target_uuid"], "value": round(float(e["weight"]), 3),
         "kind": e.get("kind") or "similar"}
        for e in edges
    ]
    stored = db.get_communities()
    community = {u: int(c["community"]) for c in stored for u in c["members"]}
    votes: dict = {}
    for e in edges:
        for a, b in ((e["source_uuid"], e["target_uuid"]), (e["target_uuid"], e["source_uuid"])):
            if a not in community and b in community:
                votes.setdefault(a, {}).setdefault(community[b], 0.0)
                votes[a][community[b]] += float(e["weight"])
    for uuid, scores in votes.items():
        community[uuid] = max(scores, key=scores.get)

    degree: dict = {}
    for link in links:
        for end in (link["source"], link["target"]):
            degree[end] = degree.get(end, 0) + 1
    # what each note is, sent with the graph so hovering shows it at once; only notes without a
    # summary or description (untagged Reddit saves...) need their text read
    customs = {u: json.loads(d.get("metadata_json") or "{}").get("custom") or {} for u, d in details.items()}
    briefs = {u: note_brief(custom) for u, custom in customs.items()}
    missing = [u for u, b in briefs.items() if not b]
    for start in range(0, len(missing), 500):
        where = "uuid IN (" + ", ".join(_q(u) for u in missing[start:start + 500]) + ")"
        for r in db.notes.search().where(where).select(["uuid", "metadata_json", "full_text"]).limit(100_000).to_list():
            briefs[r["uuid"]] = note_brief(json.loads(r["metadata_json"] or "{}").get("custom") or {}, r["full_text"] or "")
    nodes = [
        {
            "id": n["uuid"],
            "name": n["title"],
            "source_type": n["source_type"],
            "val": 1 + degree.get(n["uuid"], 0),  # node size: more links, bigger node
            "community": community.get(n["uuid"]),
            "brief": briefs.get(n["uuid"], ""),
            "image": has_image(n["source_type"], customs.get(n["uuid"], {})),  # /notes/{uuid}/thumb
            **_graph_facets(n, details.get(n["uuid"], {})),
        }
        for n in notes
    ]
    sizes: dict = {}
    for n in nodes:
        if n["community"] is not None:
            sizes[n["community"]] = sizes.get(n["community"], 0) + 1
    communities = [
        {"id": int(c["community"]), "label": c["label"], "size": sizes.get(int(c["community"]), 0)}
        for c in stored if sizes.get(int(c["community"]))
    ]
    communities.sort(key=lambda c: -c["size"])
    return {"nodes": nodes, "links": links, "communities": communities}


@app.post("/graph/rebuild")
async def rebuild_graph_endpoint(token: str = Query(...)):
    """Recompute all graph edges."""
    from .linker import rebuild_graph

    return {"status": "rebuilt", "edges": rebuild_graph()}


# --- Podcast Endpoints ---

class PodcastImportRequest(BaseModel):
    feed_url: str
    limit: int = 5


@app.get("/podcast/status")
async def get_podcast_status(token: str = Query(...)):
    """Get podcast queue status summary."""
    try:
        from .podcast.queue import TranscriptionQueue
        queue = TranscriptionQueue()
        status = queue.get_status_summary()

        # Build items list with queue position for pending items
        items_list = []
        pending_position = 1
        for item in queue.items.values():
            entry = {
                "job_id": item.episode.id,
                "podcast_name": item.episode.podcast_name or "Unknown",
                "episode_title": item.episode.episode_title or "Untitled",
                "status": item.status,
                "added_at": item.added_at.isoformat(),
                "error_message": item.error_message,
                "attempts": item.attempts,
                "audio_path": item.episode.audio_path,
            }
            if item.status == "pending":
                entry["queue_position"] = pending_position
                pending_position += 1
            items_list.append(entry)

        # Sort: pending by queue_position, then rest by added_at descending
        items_list.sort(key=lambda x: x["added_at"], reverse=True)

        return {
            "by_status": status["by_status"],
            "total": status["total"],
            "estimated_remaining": status["estimated_remaining"],
            "items": items_list,
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Podcast module not installed")
    except Exception as e:
        logger.error(f"Failed to get podcast status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/podcast/import")
async def import_podcast_feed(
    body: PodcastImportRequest,
    token: str = Query(...),
):
    """Import episodes from an RSS feed URL."""
    try:
        from .podcast.queue import TranscriptionQueue
        queue = TranscriptionQueue()
        
        job_ids = queue.add_from_feed(body.feed_url, limit=body.limit)
        
        return {
            "status": "success",
            "added_count": len(job_ids),
            "job_ids": job_ids,
            "message": f"Added {len(job_ids)} episodes to queue",
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Podcast module not installed")
    except Exception as e:
        logger.error(f"Failed to import feed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/podcast/retry")
async def retry_failed_podcasts(token: str = Query(...)):
    """Retry all failed episodes marked for review."""
    try:
        from .podcast.queue import TranscriptionQueue
        queue = TranscriptionQueue()
        
        count = queue.retry_failed()
        
        return {
            "status": "success",
            "retried_count": count,
            "message": f"Reset {count} failed items to pending",
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Podcast module not installed")
    except Exception as e:
        logger.error(f"Failed to retry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/podcast/sync")
async def sync_podcast_queue(token: str = Query(...)):
    """Sync queue with storage - finds unprocessed podcasts."""
    try:
        from .podcast.queue import TranscriptionQueue
        queue = TranscriptionQueue()
        
        stats = queue.sync_with_storage()
        
        return {
            "status": "success",
            "sync_stats": stats,
            "message": f"Sync complete: {stats.get('items_added', 0)} new items added",
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Podcast module not installed")
    except Exception as e:
        logger.error(f"Failed to sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/podcast/reset-stuck")
async def reset_stuck_podcasts(token: str = Query(...)):
    """Reset stuck processing items to pending."""
    try:
        from .podcast.queue import TranscriptionQueue
        queue = TranscriptionQueue()
        
        count = queue.reset_all_stuck()
        
        return {
            "status": "success",
            "reset_count": count,
            "message": f"Reset {count} stuck items to pending",
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Podcast module not installed")
    except Exception as e:
        logger.error(f"Failed to reset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Global processing state (read by API threads, written by background thread)
_processing_state = {
    "is_running": False,
    "current_episode": None,
    "current_episode_id": None,
    "current_stage": None,       # chunking|transcribing|diarizing|aligning|enhancing|saving
    "current_chunk": 0,
    "total_chunks": 0,
    "device": None,              # primary device (cuda|cpu)
    "chunk_device": None,        # actual device for current chunk (may differ on fallback)
    "processed_count": 0,
    "error_count": 0,
    "started_at": None,
    "episode_started_at": None,
}

# Thread-safe cancellation mechanism
_cancel_lock = threading.Lock()
_cancel_episode_id: str | None = None


def _clear_episode_state():
    """Reset per-episode fields in processing state."""
    _processing_state["current_episode"] = None
    _processing_state["current_episode_id"] = None
    _processing_state["current_stage"] = None
    _processing_state["current_chunk"] = 0
    _processing_state["total_chunks"] = 0
    _processing_state["chunk_device"] = None
    _processing_state["episode_started_at"] = None


def _run_transcription_background(limit: int = None):
    """Background task: run the podcast queue, publishing progress to _processing_state."""
    global _cancel_episode_id
    from .pipeline import run_podcast_queue

    _processing_state.update(
        is_running=True,
        started_at=datetime.utcnow().isoformat(),
        processed_count=0,
        error_count=0,
        device="cuda",
    )

    def on_start(episode_id: str, title: str) -> None:
        _processing_state.update(
            current_episode=title,
            current_episode_id=episode_id,
            current_stage="chunking",
            current_chunk=0,
            total_chunks=0,
            chunk_device="cuda",
            episode_started_at=datetime.utcnow().isoformat(),
        )

    def progress_for(episode_id: str):
        def callback(stage: str, chunk: int, total: int, device: str) -> bool:
            _processing_state["current_stage"] = stage
            if total:
                _processing_state["current_chunk"] = chunk
                _processing_state["total_chunks"] = total
            with _cancel_lock:
                return _cancel_episode_id != episode_id
        return callback

    def on_done(outcome) -> None:
        global _cancel_episode_id
        if outcome.status == "completed":
            _processing_state["processed_count"] += 1
        elif outcome.status == "failed":
            _processing_state["error_count"] += 1
        else:
            with _cancel_lock:
                _cancel_episode_id = None
        _clear_episode_state()

    try:
        run_podcast_queue(
            limit=limit,
            on_episode_start=on_start,
            on_episode_done=on_done,
            progress_for=progress_for,
            source="api",
        )
    except Exception as e:
        logger.exception(f"Background transcription error: {e}")
    finally:
        _processing_state["is_running"] = False
        _clear_episode_state()


class PodcastRunRequest(BaseModel):
    limit: int = 5


@app.post("/podcast/run")
async def run_podcast_processing(
    body: PodcastRunRequest,
    background_tasks: BackgroundTasks,
    token: str = Query(...),
):
    """Start podcast transcription processing in the background."""
    global _processing_state

    if settings.is_primary:
        from .podcast.queue import TranscriptionQueue

        if TranscriptionQueue().get_pending_count() == 0:
            return {"status": "no_pending", "message": "No pending episodes to process"}
        return _work_for_worker("podcasts")

    if _processing_state["is_running"]:
        return {
            "status": "already_running",
            "current_episode": _processing_state["current_episode"],
            "processed_count": _processing_state["processed_count"],
            "message": "Transcription is already running",
        }
    from . import runstate

    other = runstate.active_run()
    if other:
        return {
            "status": "already_running",
            "current_episode": other.get("episode_title"),
            "processed_count": other.get("completed", 0),
            "message": f"A run is already going ({other.get('source')}, pid {other.get('pid')})",
        }
    if _image_processing_state["is_running"]:
        # pass 1 stops the LLM server the image run is reading with
        return {"status": "refused", "message": "Images are being read: stop that first (or let it finish)"}

    try:
        from .podcast.queue import TranscriptionQueue
        queue = TranscriptionQueue()
        pending = queue.get_pending_count()
        
        if pending == 0:
            return {
                "status": "no_pending",
                "message": "No pending episodes to process",
            }
        
        # Start background task
        background_tasks.add_task(
            _run_transcription_background,
            limit=body.limit,
        )
        
        return {
            "status": "started",
            "pending_count": pending,
            "limit": body.limit,
            "message": f"Started processing up to {body.limit} episodes",
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Podcast module not installed")
    except Exception as e:
        logger.error(f"Failed to start processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/podcast/running")
async def get_podcast_running_status(token: str = Query(...)):
    """Get the current running status of podcast processing."""
    return {
        "is_running": _processing_state["is_running"],
        "current_episode": _processing_state["current_episode"],
        "current_episode_id": _processing_state["current_episode_id"],
        "current_stage": _processing_state["current_stage"],
        "current_chunk": _processing_state["current_chunk"],
        "total_chunks": _processing_state["total_chunks"],
        "device": _processing_state["device"],
        "chunk_device": _processing_state["chunk_device"],
        "processed_count": _processing_state["processed_count"],
        "error_count": _processing_state["error_count"],
        "started_at": _processing_state["started_at"],
        "episode_started_at": _processing_state["episode_started_at"],
    }


@app.post("/podcast/cancel/{episode_id}")
async def cancel_podcast_episode(episode_id: str, token: str = Query(...)):
    """Cancel a specific episode - remove from pending or stop current processing."""
    global _cancel_episode_id

    try:
        from .podcast.queue import TranscriptionQueue
        queue = TranscriptionQueue()

        if settings.is_primary and workerstate.cancel_episode(episode_id):
            return {"status": "cancelling", "message": "Cancellation sent to the desktop: it takes effect "
                                                       "at the next chunk boundary"}

        # Check if this is the currently-processing episode
        if _processing_state.get("current_episode_id") == episode_id:
            with _cancel_lock:
                _cancel_episode_id = episode_id
            return {
                "status": "cancelling",
                "message": "Cancellation requested. Will take effect at next chunk boundary.",
            }

        # Otherwise, check if it's pending in queue
        if episode_id in queue.items:
            item = queue.items[episode_id]
            if item.status == "pending":
                queue.mark_cancelled(episode_id)
                return {
                    "status": "cancelled",
                    "message": f"Removed pending episode: {item.episode.episode_title}",
                }
            else:
                return {
                    "status": "not_cancellable",
                    "message": f"Episode is in '{item.status}' state and cannot be cancelled",
                }

        return {"status": "not_found", "message": "Episode not found in queue"}

    except Exception as e:
        logger.error(f"Failed to cancel episode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- System (process page) ---


@app.get("/system/status")
async def system_status(token: str = Query(...)):
    """Everything the process page shows: run, queues, GPU, LLM server, database, disk, alarms."""
    from . import system

    return await asyncio.to_thread(system.status)


class StopRequest(BaseModel):
    mode: str = "after_episode"  # after_episode | abort


@app.post("/system/run/stop")
async def system_run_stop(body: StopRequest, token: str = Query(...)):
    """Stop the run (CLI or API): after the current episode, or now (abort)."""
    from . import runstate, system

    if settings.is_primary:
        return workerstate.request_stop(body.mode)
    if body.mode == "abort":
        if _processing_state["is_running"] and _processing_state["current_episode_id"]:
            global _cancel_episode_id
            with _cancel_lock:
                _cancel_episode_id = _processing_state["current_episode_id"]
            runstate.request_stop()
            return {"status": "stopping", "message": "Cancelling the current episode, then stopping"}
        return await asyncio.to_thread(system.abort_run)
    if not runstate.active_run():
        return {"status": "not_running"}
    runstate.request_stop()
    return {"status": "stopping", "message": "The run stops after the current episode"}


@app.post("/system/run/resume")
async def system_run_resume(token: str = Query(...)):
    """Withdraw a stop request that hasn't taken effect yet (primary: also lets the worker take new work)."""
    from . import runstate

    if settings.is_primary:
        return workerstate.resume()
    runstate.clear_stop()
    return {"status": "resumed"}


@app.post("/system/wake")
async def system_wake(token: str = Query(...)):
    """Wake the worker PC now (Wake-on-LAN), whatever the wake window and earlier misses."""
    from .wake import wake_now

    if not settings.is_primary:
        return {"status": "not_primary", "message": "Only the primary wakes a worker"}
    return await asyncio.to_thread(wake_now, "manual")


class DesktopSettings(BaseModel):
    stay_on: Optional[bool] = None


@app.post("/system/desktop")
async def system_desktop(body: DesktopSettings, token: str = Query(...)):
    """Keep the worker PC on when idle (stay_on), or let it shut down again."""
    if not settings.is_primary:
        raise HTTPException(status_code=400, detail="Only the primary has a worker")
    if body.stay_on is not None:
        workerstate.set_stay_on(body.stay_on)
    return workerstate.status()


@app.post("/system/unload")
async def system_unload(token: str = Query(...)):
    """Free the GPU: stop the LLM server, unload Ollama and this process's models (not during a run)."""
    from . import system

    return await asyncio.to_thread(system.free_gpu)


@app.post("/system/reindex")
async def system_reindex(token: str = Query(...)):
    """Re-read every transcript note, re-embed it and rebuild the graph (background task)."""
    from . import system
    from .pipeline import reindex_all

    return system.start_task("reindex", lambda: {"notes": reindex_all()})


# --- Settings ---

# Settings editable from the UI -> .env variable names
EDITABLE_SETTINGS = {
    "embedding_model": "EMBEDDING_MODEL",
    "reranker_model": "RERANKER_MODEL",
    "ollama_model": "OLLAMA_MODEL",
    "vision_model": "VISION_MODEL",
    "nemo_parakeet_model": "NEMO_PARAKEET_MODEL",
    "nemo_diarization_model": "NEMO_DIARIZATION_MODEL",
    "enable_auto_tagging": "ENABLE_AUTO_TAGGING",
    "chunk_size": "CHUNK_SIZE",
    "storage_path": "STORAGE_PATH",
    "hf_token": "HF_TOKEN",
    "reddit_saved_feed": "REDDIT_SAVED_FEED",
}


class SettingsUpdate(BaseModel):
    """Settings that can be updated via the API."""
    embedding_model: Optional[str] = None
    reranker_model: Optional[str] = None
    ollama_model: Optional[str] = None
    vision_model: Optional[str] = None
    nemo_parakeet_model: Optional[str] = None
    nemo_diarization_model: Optional[str] = None
    enable_auto_tagging: Optional[bool] = None
    chunk_size: Optional[int] = None
    storage_path: Optional[str] = None
    hf_token: Optional[str] = None
    reddit_saved_feed: Optional[str] = None


def update_env_file(updates: dict) -> None:
    """Write updated values into .env, replacing existing lines or appending."""
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    pending = {
        EDITABLE_SETTINGS[k]: (str(v).lower() if isinstance(v, bool) else str(v))
        for k, v in updates.items()
    }
    out = []
    for line in lines:
        key = line.split("=", 1)[0].lstrip("# ").strip()
        if key in pending:
            out.append(f"{key}={pending.pop(key)}")
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in pending.items()]
    ENV_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")


@app.get("/settings")
async def get_settings(token: str = Query(...)):
    """Get current application settings."""
    values = {key: getattr(settings, key) for key in EDITABLE_SETTINGS}
    values["storage_path"] = str(settings.storage_path)
    values["hf_token"] = "***" if settings.hf_token else ""
    # the feed link is a login to the saved list: never sent back, read fresh (no restart needed)
    from .reddit import feed_url

    values["reddit_saved_feed"] = "***" if feed_url() else ""
    return values


@app.post("/settings")
async def update_settings_endpoint(
    body: SettingsUpdate,
    token: str = Query(...),
):
    """Save settings to .env; they take effect after a restart."""
    updates = body.model_dump(exclude_none=True)
    for secret in ("hf_token", "reddit_saved_feed"):
        updates.pop(secret, None) if updates.get(secret) == "***" else None
    if updates.get("reddit_saved_feed"):
        from .reddit import RedditError, rss_feed

        try:
            rss_feed(updates["reddit_saved_feed"])
        except RedditError as e:
            raise HTTPException(status_code=400, detail=str(e))
    if not updates:
        raise HTTPException(status_code=400, detail="No settings to update")
    update_env_file(updates)
    return {
        "status": "updated",
        "updated_keys": list(updates.keys()),
        "message": "Settings saved. Restart the server for changes to take effect.",
    }


# --- Image Processing Endpoints ---

@app.get("/image/status")
async def get_image_status(token: str = Query(...)):
    """Get image queue status summary."""
    try:
        from .image.queue import ImageQueue
        queue = ImageQueue()
        status = queue.get_status_summary()

        # Get list of recent items
        items_list = []
        for item in list(queue.items.values())[-20:]:  # Last 20
            items_list.append({
                "job_id": item.image.id,
                "image_title": item.image.image_title or "Untitled",
                "collection_name": item.image.collection_name or "None",
                "status": item.status,
                "added_at": item.added_at.isoformat(),
                "error_message": item.error_message,
                "attempts": item.attempts,
                "megapixels": item.image.megapixels,
                "image_path": item.image.image_path,
            })

        # Sort by added_at descending
        items_list.sort(key=lambda x: x["added_at"], reverse=True)

        return {
            "by_status": status["by_status"],
            "total": status["total"],
            "estimated_remaining": status["estimated_remaining"],
            "items": items_list,
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Image module not installed")
    except Exception as e:
        logger.error(f"Failed to get image status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ImageScanRequest(BaseModel):
    path: str
    recursive: bool = True
    extensions: list[str] = ["png", "jpg", "jpeg", "gif", "webp", "bmp"]


@app.post("/image/scan")
async def scan_images(
    body: ImageScanRequest,
    token: str = Query(...),
):
    """Scan directory to preview images without adding to queue."""
    try:
        from pathlib import Path as P

        path = P(body.path)

        if not path.exists():
            raise HTTPException(status_code=404, detail="Path not found")

        if not path.is_dir():
            raise HTTPException(status_code=400, detail="Path must be a directory")

        # Scan for images
        found_images = []
        pattern = "**/*" if body.recursive else "*"

        for ext in body.extensions:
            for image_file in path.glob(f"{pattern}.{ext}"):
                if image_file.is_file():
                    found_images.append({
                        "path": str(image_file),
                        "name": image_file.name,
                        "size_mb": round(image_file.stat().st_size / (1024 * 1024), 2),
                    })

        # Sort by name
        found_images.sort(key=lambda x: x["name"])

        return {
            "status": "success",
            "path": str(path),
            "total_count": len(found_images),
            "images": found_images[:100],  # Limit preview to 100
            "has_more": len(found_images) > 100,
        }
    except Exception as e:
        logger.error(f"Failed to scan directory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ImageAddRequest(BaseModel):
    path: str
    collection_name: Optional[str] = None
    recursive: bool = True
    extensions: list[str] = ["png", "jpg", "jpeg", "gif", "webp", "bmp"]


@app.post("/image/add")
async def add_images(
    body: ImageAddRequest,
    token: str = Query(...),
):
    """Add image(s) to processing queue."""
    try:
        from .image.queue import ImageQueue, DuplicateImageError
        from pathlib import Path as P

        queue = ImageQueue()
        path = P(body.path)

        if not path.exists():
            raise HTTPException(status_code=404, detail="Path not found")

        job_ids = []

        if path.is_file():
            # Single image
            try:
                job_id = queue.add(path, collection_name=body.collection_name)
                job_ids.append(job_id)
            except DuplicateImageError as e:
                return {
                    "status": "duplicate",
                    "message": str(e),
                    "added_count": 0,
                }
        else:
            # Directory
            job_ids = queue.add_directory(
                path,
                collection_name=body.collection_name,
                recursive=body.recursive,
                extensions=body.extensions,
            )

        return {
            "status": "success",
            "added_count": len(job_ids),
            "job_ids": job_ids,
            "message": f"Added {len(job_ids)} image(s) to queue",
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Image module not installed")
    except Exception as e:
        logger.error(f"Failed to add images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/image/retry")
async def retry_failed_images(token: str = Query(...)):
    """Retry all failed images."""
    try:
        from .image.queue import ImageQueue
        queue = ImageQueue()

        count = queue.retry_failed()

        return {
            "status": "success",
            "retried_count": count,
            "message": f"Reset {count} failed items to pending",
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Image module not installed")
    except Exception as e:
        logger.error(f"Failed to retry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Image run state, shared with the process page (system.status)
from .system import image_run as _image_processing_state  # noqa: E402


def _run_image_processing_background(limit: int = None):
    """
    Background task to run image processing (image/worker.py): notes stored in the database,
    linked by one graph rebuild at the end. A stop request (POST /image/stop) lets the images
    being read finish, starts no more, and still links what was stored.
    """
    from .image import worker as image_worker
    from .image.queue import ImageQueue
    from .linker import rebuild_graph

    queue = ImageQueue()

    def store(image, note, seconds):
        db.upsert_note(note)
        queue.mark_completed(image.id, seconds)

    image_worker.read_images(queue, store, _image_processing_state, limit=limit, link=rebuild_graph)


@app.post("/image/stop")
async def stop_image_processing(token: str = Query(...)):
    """Stop the image run: the images being read finish, no more start, what's stored is linked."""
    if settings.is_primary:
        return workerstate.stop_images()
    if not _image_processing_state["is_running"]:
        return {"status": "not_running"}
    _image_processing_state["stop_requested"] = True
    return {"status": "stopping", "message": "Finishing the images being read, then linking them"}


class ImageRunRequest(BaseModel):
    limit: int = 5


@app.post("/image/run")
async def run_image_processing(
    body: ImageRunRequest,
    background_tasks: BackgroundTasks,
    token: str = Query(...),
):
    """Start image processing in the background."""
    if settings.is_primary:
        from .image.queue import ImageQueue

        if ImageQueue().get_pending_count() == 0:
            return {"status": "no_pending", "message": "No pending images to process"}
        return _work_for_worker("images")
    if _image_processing_state["is_running"]:
        return {
            "status": "already_running",
            "current_image": _image_processing_state["current_image"],
            "processed_count": _image_processing_state["processed_count"],
            "message": "Image processing is already running",
        }

    try:
        from .image.queue import ImageQueue
        queue = ImageQueue()
        pending = queue.get_pending_count()

        if pending == 0:
            return {
                "status": "no_pending",
                "message": "No pending images to process",
            }

        # Start background task
        background_tasks.add_task(
            _run_image_processing_background,
            limit=body.limit,
        )

        return {
            "status": "started",
            "pending_count": pending,
            "limit": body.limit,
            "message": f"Started processing up to {body.limit} images",
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Image module not installed")
    except Exception as e:
        logger.error(f"Failed to start processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Screenshot triage (keep / keep with image / skip, one at a time) ---


class TriageFolderRequest(BaseModel):
    path: str


class TriageDecisionRequest(BaseModel):
    path: str
    decision: str  # keep | keep_image | skip | ignore
    region: Optional[dict] = None  # circled part: {box: [x, y, w, h], points: [[x, y], ...]} (fractions)


@app.get("/image/triage")
async def triage_pending(token: str = Query(...), limit: int = Query(20, ge=1, le=100)):
    """The next undecided screenshots in the triage folders (newest first) and counts."""
    from .image import triage
    from .image.queue import ImageQueue

    if settings.is_primary:  # the screenshots are on the worker
        result = await _worker_json("GET", "/image/triage", params={"limit": limit})
        result["processing"] = bool((workerstate.last_beat().get("images") or {}).get("is_running"))
    else:
        result = await asyncio.to_thread(triage.pending, limit)
        result["processing"] = _image_processing_state["is_running"]
    result["queue"] = ImageQueue().get_status_summary()["by_status"]
    return result


@app.post("/image/triage/folders")
async def triage_add_folder(body: TriageFolderRequest, token: str = Query(...)):
    from .image import triage

    if settings.is_primary:
        return await _worker_json("POST", "/image/triage/folders", body=body.model_dump())
    try:
        return {"folders": triage.add_folder(body.path)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/image/triage/folders")
async def triage_remove_folder(body: TriageFolderRequest, token: str = Query(...)):
    from .image import triage

    if settings.is_primary:
        return await _worker_json("DELETE", "/image/triage/folders", body=body.model_dump())
    return {"folders": triage.remove_folder(body.path)}


@app.get("/image/triage/file")
async def triage_file(request: Request, path: str = Query(...), token: str = Query(...)):
    """A screenshot from a triage folder (nothing outside those folders is served)."""
    from .image import triage

    if settings.is_primary:
        return await _worker_file("/image/triage/file", {"path": path}, request)
    file = Path(path)
    if file.suffix.lower() not in triage.IMAGE_EXTENSIONS or not file.is_file() or not triage.in_folders(file):
        raise HTTPException(status_code=404, detail="Not a screenshot in a triage folder")
    return FileResponse(file, media_type=mimetypes.guess_type(file.name)[0] or "application/octet-stream",
                        headers={"Cache-Control": "private, max-age=3600"})


@app.post("/image/triage/decide")
async def triage_decide(body: TriageDecisionRequest, token: str = Query(...)):
    from .image import triage

    if settings.is_primary:
        return await _worker_json("POST", "/image/triage/decide", body=body.model_dump())
    try:
        return await asyncio.to_thread(triage.decide, Path(body.path), body.decision, body.region)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/image/triage/undo")
async def triage_undo(body: TriageFolderRequest, token: str = Query(...)):
    """Forget the decision on a screenshot (body.path); a keep not yet processed leaves the queue."""
    from .image import triage

    if settings.is_primary:
        return await _worker_json("POST", "/image/triage/undo", body=body.model_dump())
    return {"undone": await asyncio.to_thread(triage.undo, Path(body.path))}


# --- Reddit saves ---


@app.get("/reddit/status")
async def reddit_status(token: str = Query(...)):
    """Whether a feed is set, how many saves are imported, last sync, the running sync's progress."""
    from . import reddit

    return reddit.status()


@app.post("/reddit/sync")
async def reddit_sync(token: str = Query(...), refresh: bool = Query(False), limit: Optional[int] = Query(None, ge=1),
                      enrich: bool = Query(True), threads: bool = Query(True)):
    """
    Import new Reddit saves in the background (a maintenance task: not during a podcast run).
    enrich=false: no LLM (no tags/summary until a later sync or /reddit/tag).
    threads=false: notes from the saved feed alone, no comments (refresh=true fetches them later).
    """
    from . import reddit, system

    if not reddit.feed_url():
        return {"status": "refused", "message": "Set your Reddit saved feed link in Settings first"}
    name = "reddit comments" if refresh else "reddit sync" if enrich and threads else "reddit quick sync"
    return system.start_task(name, lambda: reddit.sync(limit=limit, refresh=refresh, enrich=enrich, threads=threads))


class RubricBody(BaseModel):
    text: str


@app.get("/reddit/rubric")
async def reddit_rubric(token: str = Query(...)):
    """The rules comment curation follows (the user's, or the default)."""
    from . import curation

    text = curation.rubric()
    return {"text": text, "default": text == curation.DEFAULT_RUBRIC, "default_text": curation.DEFAULT_RUBRIC}


@app.post("/reddit/rubric")
async def reddit_rubric_save(body: RubricBody, token: str = Query(...)):
    """Store the curation rules (empty: back to the default). Used from the next curated save on."""
    from . import curation

    curation.save_rubric(body.text)
    return {"status": "saved", "default": curation.rubric() == curation.DEFAULT_RUBRIC}


@app.post("/reddit/tag")
async def reddit_tag(token: str = Query(...)):
    """LLM tags + summaries for imported saves that have none (from the database, no Reddit requests)."""
    from . import reddit, system

    if settings.is_primary:  # the worker tags every note still without tags (tagging.py)
        return _work_for_worker("tagging")
    return system.start_task("reddit tagging", reddit.tag_untagged)


@app.post("/reddit/stop")
async def reddit_stop(token: str = Query(...)):
    """Stop a Reddit sync or tagging after the current save."""
    from . import reddit

    if not reddit.progress.get("running"):
        return {"status": "idle", "message": "No Reddit sync is running"}
    reddit.request_stop()
    return {"status": "stopping", "message": "Stops after the current save"}


@app.post("/image/triage/process")
async def triage_process(background_tasks: BackgroundTasks, token: str = Query(...)):
    """Read every kept screenshot into the knowledge base (vision model; not during a podcast run)."""
    from .image.queue import ImageQueue
    from . import runstate

    if settings.is_primary:
        if ImageQueue().get_pending_count() == 0:
            return {"status": "no_pending"}
        return _work_for_worker("screenshots")
    if runstate.active_run():
        return {"status": "refused", "message": "A podcast run is using the GPU: process after it finishes"}
    if _image_processing_state["is_running"]:
        return {"status": "already_running"}
    pending = ImageQueue().get_pending_count()
    if pending == 0:
        return {"status": "no_pending"}
    background_tasks.add_task(_run_image_processing_background, limit=None)
    return {"status": "started", "pending_count": pending}


@app.get("/image/running")
async def get_image_running_status(token: str = Query(...)):
    """Get the current running status of image processing."""
    return {
        "is_running": _image_processing_state["is_running"],
        "current_image": _image_processing_state["current_image"],
        "processed_count": _image_processing_state["processed_count"],
        "error_count": _image_processing_state["error_count"],
        "started_at": _image_processing_state["started_at"],
    }
