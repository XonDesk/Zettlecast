#!/usr/bin/env python3
"""
Start Zettlecast (any OS): Ollama if this machine needs it and it isn't running, then the API and
the web UI.

    python start.py                 API on :8000, web UI on :3000
    python start.py --no-frontend   API only

Ollama is the LLM fallback (llama-server is started by the API itself, when needed); a primary
(ROLE=primary) runs no LLM, so it never starts one. Standard library only.
"""

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from zc import ROOT, run


def env_value(key: str, default: str = "") -> str:
    """A setting from the environment or the project's .env (the API reads the same file)."""
    if os.environ.get(key):
        return os.environ[key]
    try:
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == key:
                return value.split(" #")[0].strip().strip("'\"")
    except OSError:
        pass
    return default


def ollama_running(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/version", timeout=2):
            return True
    except OSError:
        return False


def ollama_program():
    """The ollama executable: on PATH, or where the Windows installer puts it."""
    found = shutil.which("ollama")
    if found:
        return found
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return str(local) if os.environ.get("LOCALAPPDATA") and local.exists() else None


def start_ollama() -> None:
    if env_value("ROLE", "standalone").lower() == "primary":
        return
    if ollama_running(env_value("OLLAMA_BASE_URL", "http://127.0.0.1:11434")):
        return
    program = ollama_program()
    if not program:
        return  # not installed: the API uses llama-server, or reports the LLM unavailable
    print("Starting Ollama...")
    # in the background, outliving this script (as the app does when started from its menu)
    detached = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS}
                if sys.platform == "win32" else {"start_new_session": True})
    subprocess.Popen([program, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, **detached)


if __name__ == "__main__":
    os.chdir(ROOT)
    start_ollama()
    sys.exit(run(["serve", *sys.argv[1:]]))
