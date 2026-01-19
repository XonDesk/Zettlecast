"""
Zettlecast FastAPI Main Application
API endpoints for ingestion, search, and note management.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import settings
from .db import db
from .models import ProcessingResult
from .parser import parse_file, parse_url
from .search import accept_link, get_suggestions_for_note, reject_link, search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Zettlecast",
    description="Digital Zettelkasten Middleware API",
    version="0.1.0",
)

# CORS for Streamlit and bookmarklet
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
    
    # Check token
    token = request.query_params.get("token") or request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != settings.api_token:
        return JSONResponse(status_code=401, content={"error": "Invalid or missing API token"})
    
    return await call_next(request)


# --- Health Check ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


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
        # Check for duplicate URL
        existing = db.get_note_by_source_path(url)
        if existing:
            logger.info(f"URL already ingested: {url}")
            return IngestResponse(
                status="duplicate",
                uuid=existing.uuid,
                title=existing.title,
                error=f"URL already exists as '{existing.title}'",
            )
        
        # Web ingestion
        logger.info(f"Ingesting URL: {url}")
        result = parse_url(url)
    else:
        # File ingestion
        logger.info(f"Ingesting file: {file.filename}")
        
        # Save to temp file
        import tempfile
        suffix = Path(file.filename).suffix if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            result = parse_file(tmp_path)
        finally:
            tmp_path.unlink()  # Clean up
    
    if result.status == "failed":
        return IngestResponse(
            status="failed",
            error=result.error_message,
        )
    
    # Save to database if note was generated
    if result.note:
        db.upsert_note(result.note)
    
    return IngestResponse(
        status=result.status,
        uuid=result.uuid,
        title=result.title,
    )


# --- Note Endpoints ---

@app.get("/notes")
async def list_notes(
    token: str = Query(...),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    """List all notes with optional filtering."""
    notes = db.list_notes(status=status, limit=limit, offset=offset)
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
        suggestions = get_suggestions_for_note(uuid)
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
    
    results = search(query=q, top_k=top_k, rerank=rerank)
    
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


# --- Graph Data ---

@app.get("/graph")
async def get_graph_data(
    token: str = Query(...),
    limit: int = Query(2000, ge=1, le=10000),
):
    """
    Get graph nodes and edges for visualization.
    
    Returns nodes (notes) and links (edges) in a format optimized
    for react-force-graph-2d.
    """
    notes = db.list_notes(limit=limit)
    
    nodes = [
        {
            "id": n["uuid"],
            "name": n["title"],
            "source_type": n["source_type"],
            "val": 1,  # node size factor
        }
        for n in notes
    ]
    
    # Get edges from graph_edges table
    edges = db.get_all_edges()
    links = [
        {
            "source": e["source_uuid"],
            "target": e["target_uuid"],
            "value": e["weight"],
        }
        for e in edges
    ]
    
    return {"nodes": nodes, "links": links}


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
        
        # Get list of recent items
        items_list = []
        for item in list(queue.items.values())[-20:]:  # Last 20
            items_list.append({
                "job_id": item.episode.id,
                "podcast_name": item.episode.podcast_name or "Unknown",
                "episode_title": item.episode.episode_title or "Untitled",
                "status": item.status,
                "added_at": item.added_at.isoformat(),
                "error_message": item.error_message,
                "attempts": item.attempts,
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


# Global processing state
_processing_state = {
    "is_running": False,
    "current_episode": None,
    "processed_count": 0,
    "error_count": 0,
    "started_at": None,
}


def _run_transcription_background(limit: int = None, backend: str = None):
    """Background task to run transcription processing."""
    global _processing_state
    
    try:
        from .podcast.queue import TranscriptionQueue
        from .podcast.enhancer import TranscriptEnhancer
        from .podcast.formatter import save_result
        from .podcast.transcriber_factory import TranscriberFactory
        from datetime import datetime
        
        _processing_state["is_running"] = True
        _processing_state["started_at"] = datetime.utcnow().isoformat()
        _processing_state["processed_count"] = 0
        _processing_state["error_count"] = 0
        
        queue = TranscriptionQueue()
        
        # Sync queue first
        queue.sync_with_storage()
        
        # Create transcriber
        transcriber = TranscriberFactory.create(backend=backend)
        enhancer = TranscriptEnhancer()
        
        processed = 0
        
        while True:
            if limit and processed >= limit:
                break
            
            item = queue.get_next_pending()
            if not item:
                break
            
            episode = item.episode
            _processing_state["current_episode"] = episode.episode_title
            
            queue.mark_started(episode.id)
            
            try:
                # Transcribe
                from pathlib import Path as P
                result = transcriber.transcribe(
                    P(episode.audio_path),
                    episode=episode,
                )
                
                # Enhance
                enhanced = enhancer.enhance(result.full_text)
                result.keywords = enhanced.get("keywords", [])
                result.sections = enhanced.get("sections", [])
                
                # Save
                output_path = save_result(result, episode, enhanced)
                queue.mark_completed(episode.id, result, output_path)
                
                processed += 1
                _processing_state["processed_count"] = processed
                
            except Exception as e:
                logger.error(f"Failed to transcribe {episode.id}: {e}")
                queue.mark_failed(episode.id, str(e), max_retries=3)
                _processing_state["error_count"] += 1
        
        _processing_state["current_episode"] = None
        
    except Exception as e:
        logger.error(f"Background transcription error: {e}")
    finally:
        _processing_state["is_running"] = False
        _processing_state["current_episode"] = None


class PodcastRunRequest(BaseModel):
    limit: int = 5
    backend: Optional[str] = None


@app.post("/podcast/run")
async def run_podcast_processing(
    body: PodcastRunRequest,
    background_tasks: BackgroundTasks,
    token: str = Query(...),
):
    """Start podcast transcription processing in the background."""
    global _processing_state
    
    if _processing_state["is_running"]:
        return {
            "status": "already_running",
            "current_episode": _processing_state["current_episode"],
            "processed_count": _processing_state["processed_count"],
            "message": "Transcription is already running",
        }
    
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
            backend=body.backend,
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
        "processed_count": _processing_state["processed_count"],
        "error_count": _processing_state["error_count"],
        "started_at": _processing_state["started_at"],
    }


# --- Settings ---

class SettingsUpdate(BaseModel):
    """Settings that can be updated via the API."""
    embedding_model: Optional[str] = None
    reranker_model: Optional[str] = None
    whisper_model: Optional[str] = None
    llm_provider: Optional[str] = None
    ollama_model: Optional[str] = None
    enable_context_enrichment: Optional[bool] = None
    chunk_size: Optional[int] = None
    storage_path: Optional[str] = None
    asr_backend: Optional[str] = None
    hf_token: Optional[str] = None


def update_env_file(updates: dict) -> bool:
    """Update .env file with new values."""
    env_path = Path(".env")
    
    if not env_path.exists():
        logger.warning(".env file not found")
        return False
    
    # Read current content
    lines = env_path.read_text().splitlines()
    
    # Map of python setting names to env var names
    setting_to_env = {
        "embedding_model": "EMBEDDING_MODEL",
        "reranker_model": "RERANKER_MODEL",
        "whisper_model": "WHISPER_MODEL",
        "llm_provider": "LLM_PROVIDER",
        "ollama_model": "OLLAMA_MODEL",
        "enable_context_enrichment": "ENABLE_CONTEXT_ENRICHMENT",
        "chunk_size": "CHUNK_SIZE",
        "storage_path": "STORAGE_PATH",
        "asr_backend": "ASR_BACKEND",
        "hf_token": "HF_TOKEN",
    }
    
    updated_keys = set()
    new_lines = []
    
    for line in lines:
        modified = False
        for setting_name, env_name in setting_to_env.items():
            if setting_name in updates and updates[setting_name] is not None:
                if line.startswith(f"{env_name}=") or line.startswith(f"# {env_name}="):
                    value = updates[setting_name]
                    # Convert bool to lowercase string
                    if isinstance(value, bool):
                        value = str(value).lower()
                    new_lines.append(f"{env_name}={value}")
                    updated_keys.add(setting_name)
                    modified = True
                    break
        if not modified:
            new_lines.append(line)
    
    # Add any new settings that weren't in the file
    for setting_name, env_name in setting_to_env.items():
        if setting_name in updates and updates[setting_name] is not None and setting_name not in updated_keys:
            value = updates[setting_name]
            if isinstance(value, bool):
                value = str(value).lower()
            new_lines.append(f"{env_name}={value}")
    
    # Write back
    env_path.write_text("\n".join(new_lines) + "\n")
    return True


@app.get("/settings")
async def get_settings(token: str = Query(...)):
    """Get current application settings."""
    return {
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "whisper_model": settings.whisper_model,
        "llm_provider": settings.llm_provider,
        "ollama_model": settings.ollama_model,
        "enable_context_enrichment": settings.enable_context_enrichment,
        "chunk_size": settings.chunk_size,
        "storage_path": str(settings.storage_path),
        "asr_backend": settings.asr_backend,
        "hf_token": "***" if settings.hf_token else "",
    }


@app.post("/settings")
async def update_settings_endpoint(
    body: SettingsUpdate,
    token: str = Query(...),
):
    """
    Update application settings.
    
    Changes are written to .env file and will take effect on next restart.
    """
    updates = body.model_dump(exclude_none=True)
    
    if not updates:
        raise HTTPException(status_code=400, detail="No settings to update")
    
    if update_env_file(updates):
        return {
            "status": "updated",
            "updated_keys": list(updates.keys()),
            "message": "Settings saved. Restart the server for changes to take effect.",
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to update .env file")


# --- Startup ---

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Zettlecast API...")
    settings.ensure_directories()
    db.connect()
    logger.info(f"Storage path: {settings.storage_path}")
    logger.info(f"API token: {settings.api_token[:8]}...")

