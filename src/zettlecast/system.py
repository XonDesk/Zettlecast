"""
Plant status for the process page (the web UI's HMI): the run, queues, GPU, LLM server,
database, disk, and the alarms derived from them. Also the controls that aren't a single
existing endpoint: stop / abort a run in any process, free the GPU, maintenance tasks.
"""

import logging
import os
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from . import paths, runstate
from .config import settings

logger = logging.getLogger(__name__)

# VRAM (MiB, of a 32 GB card) above which Windows starts spilling to system RAM: the LLM
# then drops from ~300 to ~80 tokens/s. Normal pass-2 operation sits at ~28.5 GB.
VRAM_WARN_FRACTION = 0.94
VRAM_ALARM_FRACTION = 0.975

# Maintenance task run by the API process (reindex), shown on the process page
_task_lock = threading.Lock()
task: dict = {"name": None, "running": False, "started_at": None, "finished_at": None, "result": None, "error": None}

# Image run in the API process (main._run_image_processing_background): phase reading | linking,
# total = images it set out to read, in_flight = titles being read now
image_run: dict = {"is_running": False, "phase": None, "current_image": None, "processed_count": 0,
                   "error_count": 0, "total": 0, "in_flight": [], "stop_requested": False,
                   "started_at": None, "finished_at": None}

TRIAGE_CACHE_SECONDS = 30  # counting undecided screenshots lists every file in the folders
_triage_cache: dict = {"at": 0.0, "value": None}


def gpu() -> Optional[dict]:
    """nvidia-smi readings, or None without an NVIDIA GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        ).stdout.strip().splitlines()[0]
        name, used, total, util, temp, power = [v.strip() for v in out.split(",")]

        def num(v: str) -> Optional[float]:
            try:
                return float(v)
            except ValueError:
                return None

        return {"name": name, "memory_used_mib": num(used), "memory_total_mib": num(total),
                "utilization": num(util), "temperature": num(temp), "power_w": num(power)}
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def llm_status() -> dict:
    from . import llm

    pid, vision = llm._pid_and_vision()
    server = {"installed": llm._server_exe() is not None, "pid": pid, "vision": vision,
              "health": llm._health() if pid else None}
    server["state"] = ("ready" if server["health"] == 200 else "loading" if server["health"] == 503
                       else "stopped")
    ollama = {"reachable": False, "loaded": []}
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/ps", timeout=2)
        ollama = {"reachable": True, "loaded": [m.get("name") for m in r.json().get("models", [])]}
    except (httpx.HTTPError, ValueError):
        pass
    return {"backend": settings.llm_backend, "model": settings.ollama_model, "server": server, "ollama": ollama}


def queues() -> dict:
    from .image.queue import ImageQueue
    from .podcast.queue import TranscriptionQueue

    from .podcast.queue import _worker_alive

    q = TranscriptionQueue()
    summary = q.get_status_summary()
    workers = [{"pid": i.worker_pid, "title": i.episode.episode_title or ""} for i in q.items.values()
               if i.status == "processing" and i.worker_pid and _worker_alive(i.worker_pid, i.started_at)]
    failed = [{"id": i.episode.id, "title": i.episode.episode_title or "", "error": (i.error_message or "")[:200]}
              for i in q.items.values() if i.status == "failed"][:10]
    from .podcast.asr_cache import has_asr

    # pending episodes already through pass 1: they wait for pass 2, not for transcription
    transcribed = sum(1 for i in q.items.values()
                      if i.status == "pending" and i.episode.audio_path
                      and not paths.is_worker_ref(i.episode.audio_path) and has_asr(Path(i.episode.audio_path)))
    images = ImageQueue().get_status_summary()
    return {"podcast": {"by_status": summary["by_status"], "total": summary["total"], "failed": failed,
                        "workers": workers, "transcribed": transcribed},
            "images": images}


def database() -> dict:
    from .db import db

    try:
        stats = db.stats()
        stats["communities"] = db.communities.count_rows()
        return stats
    except Exception as e:  # a table mid-rewrite: show the rest
        logger.debug(f"db stats unavailable: {e}")
        return {}


def loaded_models() -> dict:
    """GPU models held by this (API server) process."""
    from . import embeddings, pipeline

    return {"embedder": embeddings._embedder is not None, "reranker": embeddings._reranker is not None,
            "acoustic_judge": pipeline._scorer is not None}


def alarms(run: dict, gpu_info: Optional[dict], llm: dict, q: dict, disk_free_gb: float) -> list:
    """Active abnormal conditions: {id, priority (high|medium|low), text}."""
    out = []

    def add(alarm_id: str, priority: str, text: str) -> None:
        out.append({"id": alarm_id, "priority": priority, "text": text})

    if run.get("phase", "idle") != "idle" and not run.get("alive"):
        add("run-died", "high", f"Run stopped unexpectedly during {run.get('phase')} "
                                f"({run.get('episode_title') or 'between episodes'})")
    if gpu_info and gpu_info.get("memory_total_mib"):
        share = gpu_info["memory_used_mib"] / gpu_info["memory_total_mib"]
        if share >= VRAM_ALARM_FRACTION:
            add("vram", "high", f"GPU memory {share:.0%}: spilling to system RAM, the LLM slows ~4x")
        elif share >= VRAM_WARN_FRACTION:
            add("vram", "medium", f"GPU memory {share:.0%}: close to spilling to system RAM")
        if (gpu_info.get("temperature") or 0) >= 85:
            add("gpu-temp", "medium", f"GPU temperature {gpu_info['temperature']:.0f} °C")
    if run.get("alive") and run.get("phase") == "enhance" and llm["backend"] == "llama-server" \
            and llm["server"]["installed"] and llm["server"]["state"] == "stopped" and run.get("stage") not in (None, "save", "index", "link"):
        add("llm-down", "high", "LLM server is not running during analysis")
    by = q["podcast"]["by_status"]
    if by.get("failed"):
        add("failed", "medium", f"{by['failed']} episode(s) failed (Retry failed re-queues them)")
    image_failed = ((q.get("images") or {}).get("by_status") or {}).get("failed")
    if image_failed:
        add("image-failed", "low", f"{image_failed} image(s) failed (Retry failed in Images)")
    if by.get("processing") and not run.get("alive"):
        add("stuck", "medium", f"{by['processing']} episode(s) marked processing with no run (Reset stuck)")
    if disk_free_gb < 5:
        add("disk", "high", f"Only {disk_free_gb:.0f} GB free on the storage drive")
    elif disk_free_gb < 20:
        add("disk", "medium", f"{disk_free_gb:.0f} GB free on the storage drive")
    if run.get("alive") and run.get("stop_requested"):
        add("stopping", "low", "Stop requested: the run ends after the current episode")
    return out


def triage_counts() -> Optional[dict]:
    """Screenshots still to sort, decisions so far, discards waiting for the Recycle Bin (cached)."""
    import time

    from .image import triage

    if _triage_cache["value"] is not None and time.monotonic() - _triage_cache["at"] < TRIAGE_CACHE_SECONDS:
        return _triage_cache["value"]
    try:
        p = triage.pending(limit=0)
        state = triage._load()
        to_bin = sum(1 for d in state["decisions"].values()
                     if d["decision"] == "skip" and not d.get("recycled") and not d.get("recycle_failed"))
        value = {"remaining": p["remaining"], "decided": p["decided"], "folders": len(p["folders"]),
                 "to_recycle": to_bin}
    except Exception as e:  # never let it take the process page down
        logger.debug(f"triage counts unavailable: {e}")
        value = None
    _triage_cache.update(at=time.monotonic(), value=value)
    return value


def _reddit() -> Optional[dict]:
    from . import reddit

    try:
        return reddit.status()
    except Exception as e:  # never let it take the process page down
        logger.debug(f"reddit status unavailable: {e}")
        return None


NO_LLM = {"backend": "desktop off", "model": "", "server": {"installed": False, "pid": None, "vision": False,
                                                     "health": None, "state": "stopped"},
          "ollama": {"reachable": False, "loaded": []}}


def _primary_status() -> dict:
    """The process page on the primary: the worker's run, GPU and LLM (from its heartbeat), local rest."""
    from . import workerstate

    run = runstate.read()
    beat = workerstate.last_beat()
    gpu_info = beat.get("gpu")
    llm = beat.get("llm") or NO_LLM
    q = queues()
    desktop = workerstate.status()
    disk = shutil.disk_usage(settings.storage_path if settings.storage_path.exists() else ".")
    disk_free_gb = disk.free / 2**30
    worker_task = beat.get("task") or {}
    found = alarms(run, gpu_info, llm, q, disk_free_gb)
    if desktop["wake"]["gave_up"]:
        found.append({"id": "wake-failed", "priority": "medium",
                      "text": "The desktop didn't wake for waiting jobs (check Wake-on-LAN; Wake retries)"})
    if desktop["paused"] and desktop["pending"]["total"]:
        found.append({"id": "paused", "priority": "low",
                      "text": f"Paused: {desktop['pending']['total']} job(s) wait for Resume"})
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "role": "primary",
        "run": run,
        "queues": q,
        "gpu": gpu_info,
        "llm": llm,
        "models": {**loaded_models(), **{f"worker_{k}": v for k, v in (beat.get("models") or {}).items()}},
        "database": database(),
        "disk": {"free_gb": round(disk_free_gb, 1), "total_gb": round(disk.total / 2**30, 1)},
        "worker_disk": beat.get("disk"),
        "task": dict(worker_task) if worker_task.get("running") or not task["name"] else dict(task),
        "images": beat.get("images") or {**image_run, "in_flight": []},
        "triage": beat.get("triage"),
        "reddit": _reddit(),
        "desktop": desktop,
        "alarms": found,
        "limits": {"vram_warn": VRAM_WARN_FRACTION, "vram_alarm": VRAM_ALARM_FRACTION},
    }


def status() -> dict:
    if settings.is_primary:
        return _primary_status()
    run = runstate.read()
    gpu_info = gpu()
    llm = llm_status()
    q = queues()
    workers = q["podcast"]["workers"]
    if not run["alive"] and workers:
        # a run that doesn't publish its state (started by older code): the queue knows its process
        run = {"phase": "running", "alive": True, "source": "unknown", "pid": workers[0]["pid"],
               "episode_title": workers[0]["title"], "legacy": True,
               "stop_requested": runstate.stop_requested()}
    disk = shutil.disk_usage(settings.storage_path if settings.storage_path.exists() else ".")
    disk_free_gb = disk.free / 2**30
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "role": settings.role,
        "run": run,
        "queues": q,
        "gpu": gpu_info,
        "llm": llm,
        "models": loaded_models(),
        "database": database(),
        "disk": {"free_gb": round(disk_free_gb, 1), "total_gb": round(disk.total / 2**30, 1)},
        "task": dict(task),
        "images": {**image_run, "in_flight": list(image_run["in_flight"])},
        "triage": triage_counts(),
        "reddit": _reddit(),
        "alarms": alarms(run, gpu_info, llm, q, disk_free_gb),
        "limits": {"vram_warn": VRAM_WARN_FRACTION, "vram_alarm": VRAM_ALARM_FRACTION},
    }


# --- Controls ---


def abort_run() -> dict:
    """
    Stop a run now. In another process (a CLI run): end that process and its children (the
    LLM server); the queue puts its episode back to pending when it next loads. The episode
    in progress is lost; finished ones are kept.
    """
    import psutil

    run = runstate.read()
    if not run["alive"]:
        return {"status": "not_running"}
    pid = run["pid"]
    if pid == os.getpid():
        runstate.request_stop()
        return {"status": "stopping", "message": "Run in the API server: stops after the current episode"}
    try:
        proc = psutil.Process(pid)
        for child in proc.children(recursive=True):
            child.kill()
        proc.kill()
    except psutil.NoSuchProcess:
        pass
    runstate.mark_aborted()
    from .podcast.queue import TranscriptionQueue

    TranscriptionQueue()  # loading requeues the dead worker's episode
    return {"status": "aborted", "pid": pid}


def free_gpu() -> dict:
    """Unload every model this machine runs for Zettlecast (refused during a run on this machine)."""
    if runstate.active_run() and not settings.is_primary:
        return {"status": "refused", "message": "A run is going: pause or abort it first"}
    from . import embeddings, llm
    from .pipeline import release_acoustic_scorer

    llm.unload()
    release_acoustic_scorer()
    embeddings.release_models(force=True)
    return {"status": "unloaded"}


def start_task(name: str, fn) -> dict:
    """
    Run a maintenance task in a background thread (one at a time, never during a run on this
    machine; the primary's runs are on the worker's GPU).
    """
    if runstate.active_run() and not settings.is_primary:
        return {"status": "refused", "message": "A run is going: wait for it or pause it first"}
    with _task_lock:
        if task["running"]:
            return {"status": "busy", "message": f"{task['name']} is still running"}
        task.update(name=name, running=True, started_at=datetime.now().isoformat(timespec="seconds"),
                    finished_at=None, result=None, error=None)

    def work() -> None:
        try:
            task["result"] = fn()
        except Exception as e:
            logger.exception(f"{name} failed")
            task["error"] = str(e)
        finally:
            task.update(running=False, finished_at=datetime.now().isoformat(timespec="seconds"))

    threading.Thread(target=work, name=f"task-{name}", daemon=True).start()
    return {"status": "started", "task": name}
