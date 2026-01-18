"""
Zettlecast FastAPI Main Application
API endpoints for ingestion, search, and note management.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
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


# --- Settings ---

@app.get("/settings")
async def get_settings(token: str = Query(...)):
    """Get current application settings."""
    return {
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "whisper_model": settings.whisper_model,
        "llm_provider": settings.llm_provider,
        "enable_context_enrichment": settings.enable_context_enrichment,
        "chunk_size": settings.chunk_size,
        "storage_path": str(settings.storage_path),
    }


# --- Startup ---

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Zettlecast API...")
    settings.ensure_directories()
    db.connect()
    logger.info(f"Storage path: {settings.storage_path}")
    logger.info(f"API token: {settings.api_token[:8]}...")
