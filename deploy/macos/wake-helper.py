#!/usr/bin/env python3
"""
Send the Wake-on-LAN packets the API asks for when it runs in a container (WAKE_SENDER=helper).

OrbStack's containers sit in a Linux VM behind the Mac's network address translation: a broadcast
from one never reaches the LAN, so the API writes a request to <storage>/wake_requests/ and this
script, run natively by launchd whenever that folder isn't empty (QueueDirectories), sends it.
It must leave the folder empty, or launchd runs it again and again. Standard library only (the
Mac's /usr/bin/python3).

    wake-helper.py <storage>/wake_requests      installed by install-wake-helper.sh
"""

import json
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

MAX_AGE_SECONDS = 600  # a request older than this (the helper was off) is dropped, not sent late
TEMP_GRACE_SECONDS = 60  # a .tmp the API never renamed (it crashed mid-write)


def log(message: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def magic_packet(mac: str) -> bytes:
    digits = re.sub(r"[^0-9a-fA-F]", "", mac or "")
    if len(digits) != 12:
        raise ValueError(f"Not a MAC address: {mac!r}")
    return b"\xff" * 6 + bytes.fromhex(digits) * 16


def send(mac: str, broadcast: str, port: int = 9) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(magic_packet(mac), (broadcast, port))


def process(folder: Path, now: float = None, sender=send) -> int:
    """Send every request in folder and remove it. Returns how many were sent."""
    now = time.time() if now is None else now
    sent = 0
    for path in sorted(folder.glob("*.json")):
        try:
            request = json.loads(path.read_text(encoding="utf-8"))
            if now - path.stat().st_mtime > MAX_AGE_SECONDS:
                log(f"dropped {path.name}: too old to wake for now")
            else:
                sender(request["mac"], request.get("broadcast") or "255.255.255.255", int(request.get("port") or 9))
                sent += 1
                log(f"sent Wake-on-LAN to {request['mac']} via {request.get('broadcast')}")
        except Exception as e:  # a bad request must not keep the folder non-empty
            log(f"failed {path.name}: {e}")
        path.unlink(missing_ok=True)
    for path in folder.glob(".*.tmp"):
        if now - path.stat().st_mtime > TEMP_GRACE_SECONDS:
            path.unlink(missing_ok=True)
    return sent


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: wake-helper.py <storage>/wake_requests")
    process(Path(sys.argv[1]))
