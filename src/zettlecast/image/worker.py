"""
The image run: LLM_PARALLEL images are read at once (llama-server's slots), and each finished
one is stored while the others are still being read, so the GPU isn't idle between images.
Queue updates and storing stay on the calling thread.

Used by the API server (standalone: the local queue, notes stored in the database) and by the
worker (the images it claimed from the primary, notes sent back to it).
"""

import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ..config import settings

logger = logging.getLogger(__name__)


def read_images(queue, store: Callable, state: dict, limit: Optional[int] = None,
                link: Optional[Callable[[], object]] = None) -> int:
    """
    Read the queue's pending images. store(image, note, seconds) keeps each note (and marks the
    job completed); link() runs once at the end if any were stored (linking each note as it
    comes rebuilds the graph each time: ~10 s of CPU per image, against ~2 s to read it).
    state: the run's progress for the process page; state["stop_requested"] lets the images
    being read finish, starts no more, and still links what was stored. Returns the count stored.
    """
    from .image_parser import parse_image

    try:
        pending = queue.get_pending_count()
        state.update(is_running=True, phase="reading", current_image=None, processed_count=0, error_count=0,
                     total=min(pending, limit) if limit else pending, in_flight=[], stop_requested=False,
                     started_at=datetime.now().isoformat(timespec="seconds"), finished_at=None)
        processed = submitted = 0
        running = {}  # future -> (image, start time)
        slots = max(settings.llm_parallel, 1)

        def read(image):
            return parse_image(Path(image.image_path), keep_image=image.keep_image,
                               focus_region=image.focus_region)

        def finish(image, start_time, future):
            nonlocal processed
            try:
                result = future.result()
                if result.status == "success" and result.note:
                    store(image, result.note, time.time() - start_time)
                    processed += 1
                    state["processed_count"] = processed
                else:
                    queue.mark_failed(image.id, result.error_message or "Unknown error")
                    state["error_count"] += 1
            except Exception as e:
                logger.error(f"Failed to process image {image.id}: {e}")
                queue.mark_failed(image.id, str(e))
                state["error_count"] += 1

        with ThreadPoolExecutor(max_workers=slots) as pool:
            while True:
                while len(running) < slots and not (limit and submitted >= limit) and not state["stop_requested"]:
                    item = queue.get_next_pending()
                    if not item:
                        break
                    queue.mark_started(item.image.id)  # no longer pending: not handed out twice
                    running[pool.submit(read, item.image)] = (item.image, time.time())
                    submitted += 1
                state["in_flight"] = [image.image_title or image.id for image, _ in running.values()]
                state["current_image"] = state["in_flight"][0] if state["in_flight"] else None
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    image, start_time = running.pop(future)
                    finish(image, start_time, future)

        state.update(in_flight=[], current_image=None)
        if processed and link:
            state.update(phase="linking", current_image="(linking the new notes into the graph)")
            link()
        return processed

    except Exception as e:
        logger.error(f"Background image processing error: {e}")
        return 0
    finally:
        state.update(is_running=False, phase=None, current_image=None, in_flight=[],
                     finished_at=datetime.now().isoformat(timespec="seconds"))
