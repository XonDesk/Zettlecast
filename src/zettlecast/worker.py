"""
The worker (ROLE=worker): the GPU PC doing the jobs the primary queues.

    zettlecast worker        (started at boot by Task Scheduler: deploy/windows)

It claims a batch from the primary (podcasts first, then images, then notes to tag), fetches
the inputs the primary holds (uploads, feed downloads; the worker keeps them), runs the same
pipeline a standalone install runs, and sends each result back for the primary to store. A
heartbeat thread reports the run (process page) and brings back the primary's commands: stop
after the episode, abort, cancel an episode, stop the image run, pause. Idle for
WORKER_IDLE_SHUTDOWN_MINUTES with nobody signed in to Windows, it shuts the PC down; the
primary wakes it (Wake-on-LAN) when jobs come in.

The worker keeps the media (podcast audio and its saved transcriptions, screenshots), the show
glossaries, the Wikipedia lookup cache and its own copy of the transcript notes (known speakers
for the next episodes of a show). Its file server (worker_files.py) gives the primary audio and
screenshot triage.
"""

import base64
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import httpx

from . import paths
from .config import settings
from .models import NoteModel

logger = logging.getLogger(__name__)

POLL_SECONDS = 20  # between claims while there's nothing to do
RETRY_SECONDS = 30  # after the primary couldn't be reached
LIMITS = {"podcast": 10, "image": 50, "tag": 25, "insight": 20}  # per batch


class PrimaryError(RuntimeError):
    pass


class Primary:
    """HTTP client for the primary's /worker API."""

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None, client: Optional[httpx.Client] = None):
        url = (url or settings.primary_url).rstrip("/")
        token = token or settings.worker_token
        if not url or not token:
            raise PrimaryError("Set PRIMARY_URL and WORKER_TOKEN in .env (ROLE=worker)")
        self.client = client or httpx.Client(base_url=url, timeout=60)
        self.client.headers["Authorization"] = f"Bearer {token}"

    def post(self, path: str, body: Optional[dict] = None, timeout: Optional[float] = None, tries: int = 3) -> dict:
        """POST, retrying a few times when the primary can't be reached (restarting, network blip)."""
        for attempt in range(tries):
            try:
                response = self.client.post(path, json=body or {}, timeout=timeout or 60)
                response.raise_for_status()
                return response.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                status = getattr(getattr(e, "response", None), "status_code", 0)
                if attempt == tries - 1 or 400 <= status < 500:
                    raise PrimaryError(f"POST {path}: {e}") from e
                time.sleep(2 * (attempt + 1))
        raise PrimaryError(f"POST {path} failed")

    def download(self, path: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        with self.client.stream("GET", path, timeout=600) as response:
            if response.status_code != 200:
                raise PrimaryError(f"GET {path}: HTTP {response.status_code}")
            with open(tmp, "wb") as f:
                for block in response.iter_bytes(1 << 20):
                    f.write(block)
        os.replace(tmp, target)


_primary: Optional[Primary] = None


def primary() -> Primary:
    global _primary
    if _primary is None:
        _primary = Primary()
    return _primary


# --- What the heartbeat reports and brings back ---

worker_id = f"{socket.gethostname()}-{os.getpid()}-{secrets.token_hex(3)}"  # new per start: old leases lapse
commands: dict = {"stop": False, "abort": False, "cancel_episode": None, "image_stop": False,
                  "paused": False, "shutdown_ok": True}
activity: dict = {"busy": False, "kind": None}
image_state: dict = {"is_running": False, "phase": None, "current_image": None, "processed_count": 0,
                     "error_count": 0, "total": 0, "in_flight": [], "stop_requested": False,
                     "started_at": None, "finished_at": None}
task: dict = {"name": None, "running": False, "started_at": None, "finished_at": None, "result": None,
              "error": None, "done": 0, "total": 0}


def report() -> dict:
    """The heartbeat: what the process page shows about this machine."""
    from . import runstate, system

    def safe(fn, default=None):
        try:
            return fn()
        except Exception as e:  # never let one reading stop the heartbeat
            logger.debug(f"heartbeat reading failed: {e}")
            return default

    disk = safe(lambda: shutil.disk_usage(settings.storage_path))
    return {
        "worker_id": worker_id,
        "host": socket.gethostname(),
        "file_port": settings.worker_port,
        "idle_shutdown_minutes": settings.worker_idle_shutdown_minutes,
        "busy": activity["busy"],
        "kind": activity["kind"],
        "run": safe(runstate.read, {"phase": "idle", "alive": False}),
        "images": {**image_state, "in_flight": list(image_state["in_flight"])},
        "task": dict(task),
        "gpu": safe(system.gpu),
        "llm": safe(system.llm_status),
        "models": safe(system.loaded_models, {}),
        "triage": safe(system.triage_counts),
        "disk": {"free_gb": round(disk.free / 2**30, 1), "total_gb": round(disk.total / 2**30, 1)} if disk else None,
    }


def apply_commands(reply: dict) -> None:
    """Act on the primary's heartbeat reply."""
    from . import runstate

    for key in ("stop", "abort", "cancel_episode", "image_stop", "paused", "shutdown_ok"):
        if key in reply:
            commands[key] = reply[key]
    run = runstate.read()
    if (commands["stop"] or commands["abort"]) and run.get("alive") and not run.get("stop_requested"):
        runstate.request_stop()  # the run checks this between episodes
    if commands["image_stop"] and image_state["is_running"]:
        image_state["stop_requested"] = True


def start_heartbeat(stop: Optional[threading.Event] = None) -> threading.Thread:
    stop = stop or threading.Event()

    def loop() -> None:
        while True:
            try:
                apply_commands(primary().post("/worker/heartbeat", report(), timeout=15, tries=1))
            except Exception as e:
                logger.debug(f"Heartbeat not delivered: {e}")
            if stop.wait(settings.worker_heartbeat_seconds):
                return

    thread = threading.Thread(target=loop, name="worker-heartbeat", daemon=True)
    thread.start()
    return thread


# --- Inputs ---


def local_input(kind: str, item: dict, path: str) -> Path:
    """
    The job's file on this machine: a "desktop:" path names it already; an input the primary
    holds is downloaded to the same place under this machine's storage folder, and the primary
    is told to use this copy from now on (it deletes its own).
    """
    if not item["held"]:
        return paths.from_worker_ref(path)
    rel = item["rel"] or f"uploads/{paths.name(path)}"
    target = paths.from_worker_ref(rel)
    if Path(rel).is_absolute() or paths.relative_to(target, settings.storage_path) is None:
        raise PrimaryError(f"Refusing to write outside storage: {rel}")
    if not target.is_file() or target.stat().st_size == 0:
        logger.info(f"Fetching {target.name} from the primary")
        primary().download(f"/worker/input/{kind}/{item['job_id']}", target)
    primary().post(f"/worker/{kind}/{item['job_id']}/fetched", {"ref": paths.to_worker_ref(target)})
    return target


# --- Podcasts ---


def transcript_for_primary(markdown: str) -> str:
    """The note with its audio source as the primary refers to it ("desktop:podcasts/...")."""
    from .identity import parse_frontmatter, serialize_frontmatter

    meta, body = parse_frontmatter(markdown)
    source = meta.get("source")
    if not source or paths.is_worker_ref(str(source)) or str(source).startswith(("http://", "https://")):
        return markdown
    meta["source"] = paths.to_worker_ref(Path(str(source)))
    return f"---\n{serialize_frontmatter(meta).strip()}\n---\n{body}"


class RemoteTranscriptionQueue:
    """The episodes this worker claimed, with the queue methods the pipeline calls, reported to the primary."""

    def __init__(self, items: List):
        self.items = {i.episode.id: i for i in items}

    def sync_with_storage(self) -> dict:
        return {}

    def mark_started(self, job_id: str) -> None:
        primary().post(f"/worker/podcast/{job_id}/started")

    def mark_failed(self, job_id: str, error: str, max_retries: int = 3) -> None:
        self.items[job_id].status = "failed"
        primary().post(f"/worker/podcast/{job_id}/failed", {"error": error[:2000]})

    def mark_cancelled(self, job_id: str) -> None:
        self.items[job_id].status = "cancelled"
        primary().post(f"/worker/podcast/{job_id}/failed", {"cancelled": True})

    def mark_completed(self, job_id: str, result, note_path: Path) -> None:
        """Send the note: the primary writes it to its transcripts, indexes and links it."""
        self.items[job_id].status = "completed"
        primary().post(f"/worker/podcast/{job_id}/completed", {
            "filename": Path(note_path).name,
            "markdown": transcript_for_primary(Path(note_path).read_text(encoding="utf-8")),
            "processing_time_seconds": float(getattr(result, "processing_time_seconds", 0) or 0),
        }, timeout=600)

    def release(self, job_ids) -> int:
        job_ids = [j for j in job_ids if j in self.items]
        if not job_ids:
            return 0
        return int(primary().post("/worker/podcast/release", {"job_ids": job_ids}).get("released", 0))


def _cancel_check(episode_id: str) -> Callable[[str, int, int, str], bool]:
    def callback(stage: str, chunk: int, total: int, device: str) -> bool:
        return not (commands["abort"] or commands["cancel_episode"] == episode_id)
    return callback


def run_podcasts(batch: List[dict]) -> None:
    from .pipeline import RunActiveError, run_podcast_queue
    from .podcast.models import PodcastEpisode, QueueItem

    items = []
    for entry in batch:
        episode = PodcastEpisode(**entry["episode"])
        try:
            episode.audio_path = str(local_input("podcast", entry, episode.audio_path))
        except Exception as e:
            logger.error(f"Could not get the audio of {episode.episode_title}: {e}")
            primary().post(f"/worker/podcast/{episode.id}/failed", {"error": f"Audio not available: {e}"})
            continue
        if not Path(episode.audio_path).is_file():
            primary().post(f"/worker/podcast/{episode.id}/failed",
                           {"error": f"Audio not found on the worker: {episode.audio_path}"})
            continue
        items.append(QueueItem(episode=episode, status="pending"))
    if not items:
        return
    queue = RemoteTranscriptionQueue(items)
    logger.info(f"Podcast batch: {len(items)} episode(s)")
    try:
        run_podcast_queue(
            sync=False, queue=queue, store=lambda path, check_cancel: None, progress_for=_cancel_check,
            source="worker", link=lambda: primary().post("/worker/link", timeout=3600),
        )
    except RunActiveError as e:  # another local run (a CLI run on this PC): hand the jobs back
        logger.warning(str(e))
        queue.release(list(queue.items))
        time.sleep(60)
    finally:
        # anything not finished (an exception mid-run) goes back to pending on the primary
        queue.release([job for job, i in queue.items.items() if i.status == "pending"])


# --- Images ---


class RemoteImageQueue:
    """The images this worker claimed, with the queue methods image/worker.py calls."""

    def __init__(self, items: List):
        self.items = {i.image.id: i for i in items}

    def get_pending_count(self) -> int:
        return sum(i.status == "pending" for i in self.items.values())

    def get_next_pending(self):
        return next((i for i in self.items.values() if i.status == "pending"), None)

    def mark_started(self, job_id: str) -> None:
        self.items[job_id].status = "processing"  # the claim counted the attempt

    def mark_failed(self, job_id: str, error: str) -> None:
        self.items[job_id].status = "failed"
        primary().post(f"/worker/image/{job_id}/failed", {"error": error[:2000]})


def image_for_primary(note: NoteModel) -> dict:
    """The note with its paths as the primary refers to them, plus the image if it is kept with it."""
    import hashlib

    payload = {"note": None, "image_name": None, "image_b64": None}
    custom = note.metadata.custom
    source = Path(note.source_path)
    if custom.get("image_kept") and source.is_file():
        data = source.read_bytes()
        payload["image_name"] = f"{hashlib.sha256(data).hexdigest()[:12]}{source.suffix.lower()}"
        payload["image_b64"] = base64.b64encode(data).decode("ascii")
    if not paths.is_worker_ref(note.source_path):
        note.source_path = paths.to_worker_ref(source)
    if custom.get("original_path") and not paths.is_worker_ref(custom["original_path"]):
        custom["original_path"] = paths.to_worker_ref(Path(custom["original_path"]))
    payload["note"] = note.model_dump(mode="json")
    return payload


def run_images(batch: List[dict]) -> None:
    from .image import worker as image_worker
    from .image.models import ImageItem, QueueItem

    items = []
    for entry in batch:
        image = ImageItem(**entry["image"])
        try:
            image.image_path = str(local_input("image", entry, image.image_path))
        except Exception as e:
            primary().post(f"/worker/image/{image.id}/failed", {"error": f"Image not available: {e}"})
            continue
        items.append(QueueItem(image=image, status="pending"))
    if not items:
        return
    queue = RemoteImageQueue(items)

    def store(image, note, seconds):
        primary().post(f"/worker/image/{image.id}/completed", {**image_for_primary(note), "seconds": seconds},
                       timeout=300)
        queue.items[image.id].status = "completed"

    logger.info(f"Image batch: {len(items)} image(s)")
    try:
        image_worker.read_images(queue, store, image_state,
                                 link=lambda: primary().post("/worker/link", timeout=3600))
    finally:
        left = [job for job, i in queue.items.items() if i.status in ("pending", "processing")]
        if left:
            primary().post("/worker/image/release", {"job_ids": left})


# --- Notes to tag ---


def run_tags(batch: List[dict], rubric: Optional[str] = None) -> None:
    from datetime import datetime

    from . import curation, tagging

    if rubric is not None and rubric.strip() != curation.rubric().strip():
        curation.save_rubric(rubric)  # the user's curation rules live on the primary
    notes = [NoteModel.model_validate(n) for n in batch]
    task.update(name="tagging notes", running=True, started_at=datetime.now().isoformat(timespec="seconds"),
                finished_at=None, result=None, error=None, done=0, total=len(notes))
    stored = failures_in_a_row = 0
    try:
        for n, note in enumerate(notes):
            if commands["stop"] or commands["abort"] or commands["paused"]:
                primary().post("/worker/tag/release", {"uuids": [x.uuid for x in notes[n:]]})
                break
            done, error = tagging.llm_pass(note)
            if done is None:
                primary().post(f"/worker/tag/{note.uuid}", {"error": error})
                failures_in_a_row += 1
                if failures_in_a_row >= 3:  # the LLM is down: the rest would fail too
                    primary().post("/worker/tag/release", {"uuids": [x.uuid for x in notes[n + 1:]]})
                    task["error"] = error
                    break
                continue
            failures_in_a_row = 0
            primary().post(f"/worker/tag/{note.uuid}", {"note": done.model_dump(mode="json")}, timeout=300)
            stored += 1
            task["done"] = n + 1
        if stored:
            task["name"] = "linking tagged notes"
            primary().post("/worker/link", timeout=3600)
        task["result"] = {"tagged": stored}
    finally:
        task.update(running=False, finished_at=datetime.now().isoformat(timespec="seconds"))


# --- Today's insight cards ---


def run_insights(batch: List[dict]) -> None:
    """The LLM's take on the primary's insight cards (insights.generate), each sent back as it's done."""
    from datetime import datetime

    from . import insights
    from .llm import LLMError

    task.update(name="preparing insight cards", running=True, started_at=datetime.now().isoformat(timespec="seconds"),
                finished_at=None, result=None, error=None, done=0, total=len(batch))
    done = 0
    try:
        for n, card in enumerate(batch):
            if commands["stop"] or commands["abort"] or commands["paused"]:
                primary().post("/worker/insight/release", {"keys": [c["key"] for c in batch[n:]]})
                break
            try:
                primary().post(f"/worker/insight/{card['key']}", {"insight": insights.generate(card)})
                done += 1
                task["done"] = n + 1
            except LLMError as e:
                primary().post(f"/worker/insight/{card['key']}", {"error": str(e)})
                primary().post("/worker/insight/release", {"keys": [c["key"] for c in batch[n + 1:]]})
                task["error"] = str(e)
                break
        task["result"] = {"insights": done}
    finally:
        task.update(running=False, finished_at=datetime.now().isoformat(timespec="seconds"))


# --- Idle shutdown ---


def signed_in_users() -> Optional[int]:
    """Windows sessions with a user signed in (console or Remote Desktop, connected or not); None elsewhere."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class SessionInfo(ctypes.Structure):
        _fields_ = [("SessionId", wintypes.DWORD), ("pWinStationName", wintypes.LPWSTR), ("State", ctypes.c_int)]

    wtsapi = ctypes.WinDLL("wtsapi32", use_last_error=True)
    sessions, count = ctypes.POINTER(SessionInfo)(), wintypes.DWORD()
    if not wtsapi.WTSEnumerateSessionsW(None, 0, 1, ctypes.byref(sessions), ctypes.byref(count)):
        return None
    users = 0
    try:
        for i in range(count.value):
            buffer, size = wintypes.LPWSTR(), wintypes.DWORD()
            if wtsapi.WTSQuerySessionInformationW(None, sessions[i].SessionId, 5,  # WTSUserName
                                                  ctypes.byref(buffer), ctypes.byref(size)):
                if buffer.value:
                    users += 1
                wtsapi.WTSFreeMemory(buffer)
    finally:
        wtsapi.WTSFreeMemory(sessions)
    return users


class IdleShutdown:
    """Shuts the PC down after a quiet spell, if the primary allows it and nobody is signed in."""

    GRACE_SECONDS = 120  # `shutdown /a` in this window cancels it

    def __init__(self, minutes: float, users: Callable[[], Optional[int]] = signed_in_users,
                 run: Callable[[list], object] = subprocess.run):
        self.minutes = minutes
        self.users = users
        self.run = run
        self.idle_since = time.monotonic()
        self.pending = False

    def busy(self) -> None:
        self.idle_since = time.monotonic()
        self.cancel()

    def cancel(self) -> None:
        if self.pending:
            self.run(["shutdown", "/a"])
            self.pending = False
            logger.info("Idle shutdown cancelled")

    def check(self) -> bool:
        """Called while idle: True if a shutdown was started."""
        if self.minutes <= 0 or (sys.platform != "win32" and self.run is subprocess.run):
            return False
        users = self.users()
        if not commands["shutdown_ok"] or users is None or users > 0:
            self.cancel()  # someone signed in during the grace period, or the primary said stay on
            return False
        if self.pending or time.monotonic() - self.idle_since < self.minutes * 60:
            return False
        logger.info(f"Idle {self.minutes:g} min with nobody signed in: shutting down")
        self.run(["shutdown", "/s", "/t", str(self.GRACE_SECONDS), "/c", "Zettlecast worker idle"])
        self.pending = True
        return True


# --- The loop ---


def handle(batch: dict) -> None:
    kind = batch.get("kind")
    activity.update(busy=True, kind=kind)
    try:
        if kind == "podcast":
            run_podcasts(batch["items"])
        elif kind == "image":
            run_images(batch["items"])
        elif kind == "tag":
            run_tags(batch["items"], batch.get("rubric"))
        elif kind == "insight":
            run_insights(batch["items"])
    finally:
        activity.update(busy=False, kind=None)
        from . import runstate

        runstate.clear_stop()


def run(file_server: bool = True, once: bool = False) -> None:
    """Work for the primary until stopped (once: one claim, for testing)."""
    from .image import triage

    if not settings.is_worker:
        raise PrimaryError("Set ROLE=worker in .env to run the worker")
    settings.ensure_directories()
    primary()  # fails early without PRIMARY_URL / WORKER_TOKEN
    if file_server:
        from .worker_files import start_file_server

        start_file_server()
    start_heartbeat()
    idle = IdleShutdown(settings.worker_idle_shutdown_minutes)
    logger.info(f"Worker {worker_id} working for {settings.primary_url}")
    last_sweep = 0.0
    while True:
        if time.monotonic() - last_sweep > 60:  # discarded screenshots go to the Recycle Bin
            try:
                triage.sweep()
            except Exception as e:
                logger.warning(f"Screenshot recycle sweep failed: {e}")
            last_sweep = time.monotonic()
        try:
            batch = {"kind": None} if commands["paused"] else primary().post(
                "/worker/claim", {"worker_id": worker_id, "limits": LIMITS}, tries=1)
        except PrimaryError as e:
            logger.warning(f"Primary not reachable: {e}")
            batch = {"kind": None}
            time.sleep(RETRY_SECONDS)
        if batch.get("kind"):
            idle.busy()
            try:
                handle(batch)
            except Exception:
                logger.exception(f"{batch['kind']} batch failed")
                time.sleep(RETRY_SECONDS)
            idle.busy()
        else:
            commands["paused"] = bool(batch.get("paused", commands["paused"]))
            idle.check()
        if once:
            return
        if not batch.get("kind"):
            time.sleep(POLL_SECONDS)
