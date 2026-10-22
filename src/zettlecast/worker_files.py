"""
The worker's file server (ROLE=worker, port WORKER_PORT): what only this machine has, for the
primary, which forwards the web UI's requests here while the worker is on.

    GET  /files?ref=desktop:...     audio, kept screenshots (Range requests: seeking in audio)
    GET  /image/triage              screenshots still to sort, counts, folders
    GET  /image/triage/file?path=   one of them
    POST /image/triage/decide|undo, POST|DELETE /image/triage/folders

Only the primary calls it: WORKER_TOKEN is required, and only files inside the storage folder
or a triage folder are served.
"""

import asyncio
import hmac
import logging
import mimetypes
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import paths
from .config import settings

logger = logging.getLogger(__name__)

app = FastAPI(title="Zettlecast worker files", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def verify_token(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not settings.worker_token or not hmac.compare_digest(token, settings.worker_token):
        return JSONResponse(status_code=401, content={"error": "Invalid or missing worker token"})
    return await call_next(request)


@app.get("/health")
async def health():
    return {"status": "healthy", "role": "worker"}


def servable(path: Path) -> bool:
    """Inside the storage folder or a triage folder (nothing else on the PC is served)."""
    from .image import triage

    return path.is_file() and (paths.relative_to(path, settings.storage_path) is not None or triage.in_folders(path))


@app.get("/files")
async def files(ref: str = Query(...)):
    try:
        path = paths.from_worker_ref(ref)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not servable(path):
        raise HTTPException(status_code=404, detail="Not found on the worker")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                        filename=path.name)


# --- Screenshot triage (the screenshots are on this machine) ---


class TriageFolderRequest(BaseModel):
    path: str


class TriageDecisionRequest(BaseModel):
    path: str
    decision: str
    region: Optional[dict] = None


@app.get("/image/triage")
async def triage_pending(limit: int = Query(20, ge=1, le=100)):
    from .image import triage

    return await asyncio.to_thread(triage.pending, limit)


@app.post("/image/triage/folders")
async def triage_add_folder(body: TriageFolderRequest):
    from .image import triage

    try:
        return {"folders": triage.add_folder(body.path)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/image/triage/folders")
async def triage_remove_folder(body: TriageFolderRequest):
    from .image import triage

    return {"folders": triage.remove_folder(body.path)}


@app.get("/image/triage/file")
async def triage_file(path: str = Query(...)):
    from .image import triage

    file = Path(path)
    if file.suffix.lower() not in triage.IMAGE_EXTENSIONS or not file.is_file() or not triage.in_folders(file):
        raise HTTPException(status_code=404, detail="Not a screenshot in a triage folder")
    return FileResponse(file, media_type=mimetypes.guess_type(file.name)[0] or "application/octet-stream",
                        headers={"Cache-Control": "private, max-age=3600"})


@app.post("/image/triage/decide")
async def triage_decide(body: TriageDecisionRequest):
    from .image import triage

    try:
        return await asyncio.to_thread(triage.decide, Path(body.path), body.decision, body.region)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/image/triage/undo")
async def triage_undo(body: TriageFolderRequest):
    from .image import triage

    return {"undone": await asyncio.to_thread(triage.undo, Path(body.path))}


def start_file_server() -> threading.Thread:
    """Serve on all interfaces at WORKER_PORT, in a background thread."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=settings.worker_port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="worker-files", daemon=True)
    thread.start()
    logger.info(f"Worker file server on port {settings.worker_port}")
    return thread
