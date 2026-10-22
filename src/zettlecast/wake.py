"""
Wake-on-LAN for the worker (ROLE=primary).

The worker PC is normally off. When jobs wait for it (podcasts, images, notes to tag) and it
isn't online, the primary sends a magic packet to DESKTOP_MAC_ADDRESS, then waits for its
heartbeat. Inside WAKE_WINDOW only ("01:00-07:00"; empty = any time), at most every
RETRY_MINUTES, and after MAX_MISSES unanswered packets it stops trying and raises an alarm
until the worker is heard from or woken by hand (POST /system/wake).

The PC must allow it: Wake-on-LAN from S5 in the BIOS, "Wake on Magic Packet" on the network
card, Fast Startup off (deploy/windows/README.md).
"""

import logging
import re
import socket
import threading
from datetime import datetime
from datetime import time as dtime
from typing import Optional

from . import workerstate
from .config import settings

logger = logging.getLogger(__name__)

RETRY_MINUTES = 15
MAX_MISSES = 3
CHECK_SECONDS = 60


def magic_packet(mac: str) -> bytes:
    """6 x 0xFF, then the MAC address 16 times."""
    digits = re.sub(r"[^0-9a-fA-F]", "", mac or "")
    if len(digits) != 12:
        raise ValueError(f"Not a MAC address: {mac!r}")
    return b"\xff" * 6 + bytes.fromhex(digits) * 16


def send(mac: Optional[str] = None, broadcast: Optional[str] = None, port: int = 9) -> None:
    packet = magic_packet(mac or settings.desktop_mac_address)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(packet, (broadcast or settings.wake_broadcast, port))


def in_window(window: str, now: Optional[datetime] = None) -> bool:
    """Whether now falls in "HH:MM-HH:MM" (may wrap past midnight; empty = always)."""
    if not window or not window.strip():
        return True
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", window)
    if not match:
        logger.warning(f"WAKE_WINDOW not understood ({window!r}): waking at any time")
        return True
    h1, m1, h2, m2 = map(int, match.groups())
    start, end, t = dtime(h1, m1), dtime(h2, m2), (now or datetime.now()).time()
    return start <= t < end if start <= end else t >= start or t < end


def wake_now(reason: str = "manual") -> dict:
    """Send the packet now (the Wake button: ignores the window, retries and earlier misses)."""
    if not settings.desktop_mac_address:
        return {"status": "not_configured", "message": "Set DESKTOP_MAC_ADDRESS on the primary to wake the worker"}
    if workerstate.online():
        return {"status": "online", "message": "The worker is already on"}
    try:
        send()
    except (OSError, ValueError) as e:
        logger.error(f"Wake-on-LAN failed: {e}")
        return {"status": "failed", "message": str(e)}
    workerstate.update_wake(sent_at=datetime.now().isoformat(timespec="seconds"), reason=reason,
                            **({"misses": 0, "gave_up": False} if reason == "manual" else {}))
    logger.info(f"Wake-on-LAN sent to {settings.desktop_mac_address} ({reason})")
    return {"status": "sent", "message": "Wake-up sent: the worker should report in a minute or two"}


def tick(now: Optional[datetime] = None) -> Optional[str]:
    """
    One look by the policy: what it did ("sent", "gave_up") or None. Waking for jobs only when
    configured, not paused, in the window, the worker is offline and jobs are waiting.
    """
    now = now or datetime.now()
    if not settings.desktop_mac_address or workerstate.online():
        return None
    state = workerstate.load()
    wake = state["wake"]
    if state["paused"] or wake["gave_up"] or not in_window(settings.wake_window, now):
        return None
    pending = workerstate.pending_work()
    if not pending["total"]:
        return None
    if wake["sent_at"]:
        if (now - datetime.fromisoformat(wake["sent_at"])).total_seconds() < RETRY_MINUTES * 60:
            return None  # booting, or the packet got lost: wait before counting it as missed
        misses = wake["misses"] + 1
        if misses >= MAX_MISSES:
            workerstate.update_wake(misses=misses, gave_up=True, sent_at=None)
            logger.warning(f"The worker didn't wake after {misses} tries: not trying again until it is "
                           f"seen or woken by hand")
            return "gave_up"
        workerstate.update_wake(misses=misses)
    parts = [f"{n} {kind}" for kind, n in pending.items() if kind != "total" and n]
    result = wake_now(reason="jobs waiting: " + ", ".join(parts))
    return "sent" if result["status"] == "sent" else None


def start_policy(stop: Optional[threading.Event] = None, every: float = CHECK_SECONDS) -> threading.Thread:
    stop = stop or threading.Event()

    def loop() -> None:
        while not stop.wait(every):
            try:
                tick()
            except Exception as e:  # never let it die: the next look may work
                logger.warning(f"Wake policy check failed: {e}")

    thread = threading.Thread(target=loop, name="wake-policy", daemon=True)
    thread.start()
    return thread
