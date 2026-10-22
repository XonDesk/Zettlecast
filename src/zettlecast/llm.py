"""
LLM client shared by every LLM step (transcript enhancement, tagging, image analysis).

Two backends serve the same model (settings.ollama_model):
- llama-server (llama.cpp), started on demand with the GGUF and vision projector from
  Ollama's own model store: parallel slots + speculative decoding, ~4x faster cleanup.
  Stopped by unload(), after llm_idle_minutes unused, and when the starting process exits.
- Ollama (/api/chat): fallback when llama-server.exe isn't installed, and for any other model.

Thinking is disabled and the context window is explicit. Pass a JSON schema to get
structured output back as a dict.
"""

import atexit
import base64
import json
import logging
import mimetypes
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

import httpx

from .config import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM is unreachable or returns an unusable response."""


def chat(
    prompt: str,
    schema: Optional[dict] = None,
    images: Optional[list[Path]] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.2,
) -> Any:
    """
    Run one chat turn.

    Args:
        prompt: User message
        schema: JSON schema; when given, the response is parsed and returned as a dict
        images: Image files to attach (vision)
        model: Override settings.ollama_model (another model always goes to Ollama)
        system: Optional system message
        temperature: Sampling temperature

    Returns:
        Response text, or the parsed JSON object when schema is given
    """
    if settings.is_primary:
        # no GPU on the primary: LLM work is queued for the worker (which claims it over HTTP)
        raise LLMError("No LLM on the primary machine (ROLE=primary): the worker does this")
    if _use_server(model):
        content = _server_chat(prompt, schema, images, system, temperature)
    else:
        content = _ollama_chat(prompt, schema, images, model, system, temperature)
    if not schema:
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMError(f"Model returned invalid JSON: {content[:200]}") from e


def _ollama_chat(prompt, schema, images, model, system, temperature) -> str:
    user_msg: dict = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = [base64.b64encode(Path(p).read_bytes()).decode() for p in images]

    messages = [{"role": "system", "content": system}] if system else []
    messages.append(user_msg)

    body: dict = {
        "model": model or settings.ollama_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"num_ctx": settings.llm_num_ctx, "temperature": temperature},
    }
    if settings.llm_seed is not None:
        body["options"]["seed"] = settings.llm_seed
    if schema:
        body["format"] = schema

    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json=body,
            timeout=settings.llm_timeout_seconds,
        )
    except httpx.ConnectError as e:
        raise LLMError(f"Cannot connect to Ollama at {settings.ollama_base_url}") from e

    if response.status_code == 404:
        raise LLMError(f"Model '{body['model']}' not found. Run: ollama pull {body['model']}")
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


# --- llama-server backend ---

_server_lock = threading.Lock()
_last_used = time.monotonic()
_idle_thread: Optional[threading.Thread] = None
_own_pid: Optional[int] = None  # the llama-server this process started
_warned_missing = False


def _server_exe() -> Optional[Path]:
    if settings.llama_server_exe:
        exe = Path(settings.llama_server_exe)
        return exe if exe.exists() else None
    # newest build under %LOCALAPPDATA%\llama.cpp\<build>\ (b11399, b11400, ... sort by number)
    root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "llama.cpp"
    found = sorted(root.glob("*/llama-server.exe"),
                   key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))
    return found[-1] if found else None


def _use_server(model: Optional[str]) -> bool:
    global _warned_missing
    if settings.llm_backend != "llama-server" or (model and model != settings.ollama_model):
        return False
    if _server_exe() is None:
        if not _warned_missing:
            logger.warning("llama-server.exe not found; using Ollama (see LLAMA_SERVER_EXE)")
            _warned_missing = True
        return False
    return True


def ollama_blob(model: str, media_type: str) -> Optional[Path]:
    """Path of one layer ('model', 'projector') of an Ollama model in Ollama's store."""
    store = Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")
    name, _, tag = model.partition(":")
    if "/" not in name:
        name = f"library/{name}"
    manifest = store / "manifests" / "registry.ollama.ai" / name / (tag or "latest")
    if not manifest.exists():
        return None
    for layer in json.loads(manifest.read_text(encoding="utf-8")).get("layers", []):
        if layer.get("mediaType") == f"application/vnd.ollama.image.{media_type}":
            blob = store / "blobs" / layer["digest"].replace(":", "-")
            return blob if blob.exists() else None
    return None


def _server_url() -> str:
    return f"http://127.0.0.1:{settings.llama_server_port}"


def _pid_file() -> Path:
    return settings.storage_path / ".llama-server.pid"


def _health() -> Optional[int]:
    """HTTP status of /health (200 = ready, 503 = loading), None when nothing listens."""
    try:
        return httpx.get(f"{_server_url()}/health", timeout=2).status_code
    except httpx.HTTPError:
        return None


def _pid_and_vision() -> tuple[Optional[int], bool]:
    """(pid, started with the vision projector) of the running llama-server, from its pid file."""
    try:
        pid, _, vision = _pid_file().read_text().strip().partition(" ")
        return int(pid), vision == "vision"
    except (OSError, ValueError):
        return None, False


def _serves_images() -> bool:
    """Whether the running llama-server takes images: asked from the server (its pid file can be gone)."""
    try:
        return bool(httpx.get(f"{_server_url()}/props", timeout=2).json()["modalities"]["vision"])
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return _pid_and_vision()[1]


def _port_pid() -> Optional[int]:
    """Pid of the process listening on llama-server's port (for a server whose pid file is gone)."""
    try:
        import psutil

        for c in psutil.net_connections(kind="tcp"):
            if c.laddr and c.laddr.port == settings.llama_server_port and c.status == psutil.CONN_LISTEN:
                if c.pid and psutil.Process(c.pid).name().lower().startswith("llama-server"):
                    return c.pid
    except Exception as e:
        logger.debug(f"Port owner lookup failed: {e}")
    return None


def _ensure_server(vision: bool = False) -> None:
    """
    Start llama-server unless it's already up (perhaps started by another process).
    The vision projector (~1 GB) is loaded only for image calls: a text run needs the room
    for the acoustic judge and the embedding model (past 32 GB, Windows spills VRAM to
    system RAM and generation drops from ~300 to ~80 tokens/s).
    """
    global _idle_thread, _own_pid
    with _server_lock:
        status = _health()
        if status == 200:
            if not vision or _serves_images():
                return
            stop_server()  # restart with the projector
            status = _health()
            if status is not None:
                raise LLMError("llama-server without the vision projector is running and could not be stopped")
        log = settings.storage_path / "llama-server.log"
        if status is None:
            model = ollama_blob(settings.ollama_model, "model")
            if model is None:
                raise LLMError(f"{settings.ollama_model} not in Ollama's store. Run: ollama pull {settings.ollama_model}")
            cmd = [str(_server_exe()), "-m", str(model), "-ngl", "999",
                   # one context pool shared by the slots: the long analysis call and a
                   # few cleanup chunks fit together (4 separate 32K slots cost 1.3 GB more)
                   "-c", str(settings.llama_server_ctx), "--kv-unified",
                   "-np", str(max(settings.llm_parallel, 1)),
                   "--host", "127.0.0.1", "--port", str(settings.llama_server_port),
                   *shlex.split(settings.llama_server_args)]
            projector = ollama_blob(settings.ollama_model, "projector") if vision else None
            if projector:
                cmd += ["--mmproj", str(projector)]
            logger.info(f"Starting llama-server for {settings.ollama_model}")
            settings.storage_path.mkdir(parents=True, exist_ok=True)
            with open(log, "wb") as out:
                process = subprocess.Popen(
                    cmd, stdout=out, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
            _pid_file().write_text(f"{process.pid} vision" if projector else str(process.pid))
            _own_pid = process.pid
            atexit.register(stop_server, only_own=True)
            if _idle_thread is None and settings.llm_idle_minutes > 0:
                _idle_thread = threading.Thread(target=_idle_loop, name="llm-idle-unloader", daemon=True)
                _idle_thread.start()
        else:
            process = None  # loading, started elsewhere
        deadline = time.monotonic() + settings.llm_timeout_seconds
        while time.monotonic() < deadline:
            if _health() == 200:
                return
            if process is not None and process.poll() is not None:
                tail = log.read_text(encoding="utf-8", errors="replace")[-600:] if log.exists() else ""
                raise LLMError(f"llama-server exited while loading:\n{tail}")
            time.sleep(1)
        raise LLMError("llama-server did not become ready in time")


def stop_server(only_own: bool = False) -> None:
    """
    Stop the llama-server (whichever process started it, so a run can free the GPU);
    only_own: only if this process started it (exit, idle timer).
    """
    pid_file = _pid_file()
    if pid_file.exists():
        pid, _ = _pid_and_vision()
    elif not only_own and _health() is not None:
        pid = _port_pid()  # up without a pid file
    else:
        return
    try:
        if pid is None or (only_own and pid != _own_pid):
            return
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, 15)
    except (ValueError, OSError):
        pass
    pid_file.unlink(missing_ok=True)
    # wait for the port and VRAM to be released
    for _ in range(30):
        if _health() is None:
            break
        time.sleep(0.5)


def _idle_loop() -> None:
    while True:
        time.sleep(30)
        if time.monotonic() - _last_used > settings.llm_idle_minutes * 60 and _pid_file().exists():
            with _server_lock:
                if time.monotonic() - _last_used > settings.llm_idle_minutes * 60:
                    stop_server(only_own=True)


def _server_chat(prompt, schema, images, system, temperature) -> str:
    global _last_used
    _ensure_server(vision=bool(images))
    _last_used = time.monotonic()

    content: Any = prompt
    if images:
        content = [{"type": "text", "text": prompt}]
        for p in images:
            mime = mimetypes.guess_type(str(p))[0] or "image/png"
            data = base64.b64encode(Path(p).read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": content})

    body: dict = {
        "messages": messages,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if settings.llm_seed is not None:
        body["seed"] = settings.llm_seed
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}

    try:
        try:
            response = httpx.post(f"{_server_url()}/v1/chat/completions", json=body,
                                  timeout=settings.llm_timeout_seconds)
        except (httpx.ConnectError, httpx.RemoteProtocolError):
            # stopped under us (another process's run or idle timer): start it again, once
            _ensure_server(vision=bool(images))
            response = httpx.post(f"{_server_url()}/v1/chat/completions", json=body,
                                  timeout=settings.llm_timeout_seconds)
    except httpx.HTTPError as e:
        raise LLMError(f"llama-server request failed: {e}") from e
    finally:
        _last_used = time.monotonic()
    if response.status_code != 200:
        raise LLMError(f"llama-server returned {response.status_code}: {response.text[:300]}")
    return response.json()["choices"][0]["message"]["content"].strip()


def chat_each(calls: Iterable[Callable[[], Any]]) -> Iterator[Any]:
    """
    Run independent LLM calls settings.llm_parallel at a time and yield each result, or the
    exception it raised, in order. Results are yielded as soon as they're ready, so the caller
    can work on the first while the rest generate. llama-server generates them together
    (parallel slots); Ollama queues them (it can't batch qwen35moe).
    """
    def guarded(call: Callable[[], Any]) -> Any:
        try:
            return call()
        except Exception as e:  # handed to the caller with the others' results
            return e

    calls = list(calls)
    if settings.llm_parallel <= 1 or len(calls) <= 1:
        yield from (guarded(c) for c in calls)
        return
    with ThreadPoolExecutor(max_workers=settings.llm_parallel) as pool:
        yield from pool.map(guarded, calls)


def unload(model: Optional[str] = None, only_own: bool = False) -> None:
    """
    Free the LLM's VRAM now (before GPU-heavy ASR work): stop llama-server, unload Ollama.
    only_own: leave a llama-server another process started (shutting down the API server).
    """
    if settings.is_primary:
        return
    stop_server(only_own=only_own)
    try:
        httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": model or settings.ollama_model, "keep_alive": 0},
            timeout=30,
        )
    except httpx.HTTPError as e:
        logger.debug(f"Ollama unload skipped: {e}")
