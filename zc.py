#!/usr/bin/env python3
"""
Zettlecast CLI shortcut (any OS): runs the CLI installed in the project's .venv.

    python zc.py podcast run
    python zc.py search "gut training"

Standard library only, so any Python 3 can run it.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cli() -> Path:
    """The zettlecast command in .venv (Scripts\\zettlecast.exe on Windows, bin/zettlecast elsewhere)."""
    for path in (ROOT / ".venv" / "Scripts" / "zettlecast.exe", ROOT / ".venv" / "bin" / "zettlecast"):
        if path.exists():
            return path
    sys.exit(f"No Zettlecast install in {ROOT / '.venv'}: create the venv and install the project first (README).")


def run(args: list) -> int:
    """Run the CLI with args; Ctrl+C reaches it too, and it gets to shut down (servers stop their children)."""
    proc = subprocess.Popen([str(cli()), *args], cwd=ROOT)
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            continue  # the CLI got the same Ctrl+C: wait for it to finish


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(run(sys.argv[1:]))
