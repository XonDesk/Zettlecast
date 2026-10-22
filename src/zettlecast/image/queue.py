"""
Image Processing Queue Manager

Persistent queue for batch image processing with retry logic and time estimation.
Mirrors podcast/queue.py pattern but adapted for images with megapixel-based ETA.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from PIL import Image

from .. import statefile
from ..config import settings
from .models import ImageItem, QueueItem

logger = logging.getLogger(__name__)


class DuplicateImageError(Exception):
    """Raised when attempting to add a duplicate image."""
    pass


class ImageQueue:
    """
    Persistent queue for batch image processing.

    State is saved to JSON file for crash recovery.
    Tracks processing times per megapixel for accurate ETA estimation.
    """

    DATETIME_FIELDS = ("added_at", "started_at", "completed_at", "lease_until")
    LISTS = {"processing_times_per_mp": 100, "completed_hashes": 5000}  # entries kept

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or (settings.storage_path / "image_queue.json")
        self.items: Dict[str, QueueItem] = {}
        self.processing_times_per_mp: List[float] = []  # Seconds per megapixel
        self.completed_hashes: List[str] = []
        self._base: dict = {}  # the file as loaded: saving merges with changes made elsewhere since

        self._load_state()

    def _load_state(self):
        """Load queue state from disk."""
        if self.state_file.exists():
            try:
                self._parse(statefile.read(self.state_file))
                logger.debug(f"Loaded queue state: {len(self.items)} items")

                # Auto-reset stuck 'processing' items
                self._reset_stuck_items()

            except Exception as e:
                logger.error(f"Failed to load queue state: {e}")
                self.items = {}

    def _parse(self, data: dict) -> None:
        self._base = json.loads(json.dumps(data))  # a copy: parsing must not change it
        self.items = {}
        for job_id, item_data in data.get("items", {}).items():
            item_data = dict(item_data)
            image = ImageItem(**item_data.pop("image", {}))
            for dt_field in self.DATETIME_FIELDS:
                if item_data.get(dt_field):
                    item_data[dt_field] = datetime.fromisoformat(item_data[dt_field])
            self.items[job_id] = QueueItem(image=image, **item_data)
        self.processing_times_per_mp = list(data.get("processing_times_per_mp", []))
        self.completed_hashes = list(data.get("completed_hashes", []))

    def _reset_stuck_items(self):
        """
        Reset items stuck in 'processing' state.

        An item claimed by a worker (another machine) is stuck once its lease runs out; others
        after more than 1 hour of 'processing' (e.g., due to process crash).
        """
        now = datetime.utcnow()
        stuck_threshold = timedelta(hours=1)
        reset_count = 0

        for item in self.items.values():
            if item.status == "processing" and item.lease_until:
                if item.lease_until < now:
                    logger.warning(f"Lease ran out, back to pending: {item.image.image_title}")
                    item.status = "pending"
                    item.started_at = item.worker_id = item.lease_until = None
                    reset_count += 1
            elif item.status == "processing":
                if item.started_at:
                    processing_time = now - item.started_at
                    if processing_time > stuck_threshold:
                        logger.warning(
                            f"Resetting stuck job (processing for {processing_time}): "
                            f"{item.image.image_title or item.image.image_path}"
                        )
                        item.status = "pending"
                        item.started_at = None
                        reset_count += 1
                else:
                    # No start time but marked as processing - reset it
                    item.status = "pending"
                    reset_count += 1

        if reset_count > 0:
            logger.info(f"Auto-reset {reset_count} stuck 'processing' items to 'pending'")
            self._save_state()

    def _serialize(self) -> dict:
        data = {
            "items": {},
            "processing_times_per_mp": list(self.processing_times_per_mp),
            "completed_hashes": list(self.completed_hashes),
        }
        for job_id, item in self.items.items():
            item_dict = item.model_dump()
            for dt_field in self.DATETIME_FIELDS:
                if item_dict.get(dt_field):
                    item_dict[dt_field] = item_dict[dt_field].isoformat()
            data["items"][job_id] = item_dict
        # the same JSON types as a file read back (the merge compares item by item)
        return json.loads(json.dumps(data, default=str))

    def _save_state(self):
        """Save queue state to disk, merged with changes other writers made since it was loaded."""
        try:
            merged = statefile.save(self.state_file, self._base, self._serialize(), list_keys=self.LISTS)
            self._parse(merged)
        except Exception as e:
            logger.error(f"Failed to save queue state: {e}")

    @staticmethod
    def compute_image_hash(image_path: Path) -> str:
        """Compute SHA256 hash of image file."""
        sha256 = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def add(
        self,
        image_path: Path,
        collection_name: Optional[str] = None,
        skip_duplicate: bool = True,
    ) -> str:
        """
        Add an image to the queue.

        Args:
            image_path: Path to image file
            collection_name: Optional collection/folder name
            skip_duplicate: Whether to skip if already processed

        Returns:
            Job ID

        Raises:
            DuplicateImageError: If image already processed and skip_duplicate=True
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Compute hash for deduplication
        image_hash = self.compute_image_hash(image_path)

        # Check for duplicate
        if skip_duplicate and image_hash in self.completed_hashes:
            raise DuplicateImageError(
                f"Image already processed: {image_path.name} (hash: {image_hash[:8]})"
            )

        # Check if already in queue
        for item in self.items.values():
            if item.image.image_hash == image_hash:
                logger.info(f"Image already in queue: {image_path.name}")
                return item.image.id

        # Get image dimensions for time estimation
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                megapixels = (width * height) / 1_000_000
        except Exception as e:
            logger.warning(f"Could not read dimensions for {image_path.name}: {e}")
            width = height = None
            megapixels = None

        # Create image item
        image = ImageItem(
            image_path=str(image_path.absolute()),
            image_hash=image_hash,
            collection_name=collection_name,
            image_title=image_path.stem,
            width=width,
            height=height,
            megapixels=megapixels,
        )

        # Create queue item
        item = QueueItem(
            image=image,
            status="pending",
            added_at=datetime.utcnow(),
        )

        self.items[image.id] = item
        self._save_state()

        logger.info(f"Added to queue: {image.image_title} (ID: {image.id[:8]})")
        return image.id

    def add_directory(
        self,
        directory: Path,
        collection_name: Optional[str] = None,
        recursive: bool = True,
        extensions: List[str] = None,
    ) -> List[str]:
        """
        Add all images from a directory.

        Args:
            directory: Directory to scan
            collection_name: Collection name to assign to all images
            recursive: Whether to scan subdirectories
            extensions: File extensions to include (default: png, jpg, jpeg, gif, webp, bmp)

        Returns:
            List of added job IDs
        """
        extensions = extensions or ["png", "jpg", "jpeg", "gif", "webp", "bmp"]
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        job_ids = []
        pattern = "**/*" if recursive else "*"

        for ext in extensions:
            for image_file in directory.glob(f"{pattern}.{ext}"):
                try:
                    job_id = self.add(
                        image_path=image_file,
                        collection_name=collection_name or directory.name,
                    )
                    job_ids.append(job_id)
                except DuplicateImageError:
                    logger.debug(f"Skipping duplicate: {image_file.name}")
                except Exception as e:
                    logger.error(f"Failed to add {image_file.name}: {e}")

        logger.info(f"Added {len(job_ids)} images from {directory}")
        return job_ids

    def set_options(self, job_id: str, keep_image: bool, focus_region: Optional[dict] = None) -> None:
        """
        keep_image: store a copy of the image with its note (otherwise only the information).
        focus_region: the part the user circled (the vision pass reads that as the focus).
        """
        if job_id in self.items:
            self.items[job_id].image.keep_image = keep_image
            self.items[job_id].image.focus_region = focus_region
            self._save_state()

    def add_remote(
        self,
        ref: str,
        image_hash: str,
        collection_name: Optional[str] = None,
        title: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        keep_image: bool = True,
        focus_region: Optional[dict] = None,
    ) -> str:
        """
        Queue an image that is on the worker machine (ref: "desktop:..."), described by the worker.

        Raises:
            DuplicateImageError: If the image was already processed
        """
        if image_hash in self.completed_hashes:
            raise DuplicateImageError(f"Image already processed: {title or ref} (hash: {image_hash[:8]})")
        for item in self.items.values():
            if item.image.image_hash == image_hash:
                return item.image.id
        image = ImageItem(
            image_path=ref,
            image_hash=image_hash,
            collection_name=collection_name,
            image_title=title,
            width=width,
            height=height,
            megapixels=(width * height) / 1_000_000 if width and height else None,
            keep_image=keep_image,
            focus_region=focus_region,
        )
        self.items[image.id] = QueueItem(image=image, status="pending", added_at=datetime.utcnow())
        self._save_state()
        logger.info(f"Added to queue from the worker: {image.image_title} (ID: {image.id[:8]})")
        return image.id

    def remove_by_hash(self, image_hash: str) -> Optional[bool]:
        """Take a not-yet-processed image out (None: not queued; False: already processed)."""
        item = next((i for i in self.items.values() if i.image.image_hash == image_hash), None)
        if item is None:
            return None
        return self.remove_pending(item.image.id)

    def claim(self, worker_id: str, lease: timedelta, limit: Optional[int] = None) -> List[QueueItem]:
        """Hand pending images to a worker: processing under a lease it renews with each heartbeat."""
        pending = [i for i in self.items.values() if i.status == "pending"]
        if limit is not None:
            pending = pending[:limit]
        now = datetime.utcnow()
        for item in pending:
            item.status = "processing"
            item.started_at = now
            item.worker_id = worker_id
            item.lease_until = now + lease
            item.attempts += 1
        if pending:
            self._save_state()
        return [self.items[i.image.id] for i in pending]

    def renew(self, worker_id: str, lease: timedelta) -> int:
        """Extend the leases of the images a worker holds."""
        held = [i for i in self.items.values() if i.status == "processing" and i.worker_id == worker_id]
        for item in held:
            item.lease_until = datetime.utcnow() + lease
        if held:
            self._save_state()
        return len(held)

    def release(self, job_ids) -> int:
        """Put claimed images that weren't read back to pending (the attempt isn't counted)."""
        count = 0
        for job_id in job_ids:
            item = self.items.get(job_id)
            if item and item.status == "processing":
                item.status = "pending"
                item.started_at = item.worker_id = item.lease_until = None
                item.attempts = max(0, item.attempts - 1)
                count += 1
        if count:
            self._save_state()
        return count

    def set_path(self, job_id: str, path: str) -> None:
        """Point a job at another copy of its file (the worker's, once it has fetched it)."""
        if job_id in self.items:
            self.items[job_id].image.image_path = path
            self._save_state()

    def remove_pending(self, job_id: str) -> bool:
        """Take a not-yet-processed image out of the queue."""
        item = self.items.get(job_id)
        if not item or item.status not in ("pending", "failed", "review"):
            return False
        del self.items[job_id]
        self._save_state()
        return True

    def get_next_pending(self) -> Optional[QueueItem]:
        """
        Get the next pending item to process.

        Returns:
            QueueItem or None if no pending items
        """
        for item in self.items.values():
            if item.status == "pending":
                return item
        return None

    def get_pending_count(self) -> int:
        """Get count of pending items."""
        return sum(1 for item in self.items.values() if item.status == "pending")

    def mark_started(self, job_id: str):
        """Mark an item as started processing."""
        if job_id in self.items:
            self.items[job_id].status = "processing"
            self.items[job_id].started_at = datetime.utcnow()
            self.items[job_id].attempts += 1
            self._save_state()

    def mark_completed(self, job_id: str, processing_time_seconds: float):
        """
        Mark an item as completed.

        Args:
            job_id: Job ID
            processing_time_seconds: Total processing time
        """
        if job_id in self.items:
            item = self.items[job_id]
            item.status = "completed"
            item.completed_at = datetime.utcnow()
            item.processing_time_seconds = processing_time_seconds
            item.worker_id = item.lease_until = None

            # Track processing time per megapixel for ETA
            if item.image.megapixels and item.image.megapixels > 0:
                time_per_mp = processing_time_seconds / item.image.megapixels
                self.processing_times_per_mp.append(time_per_mp)

            # Add to completed hashes
            self.completed_hashes.append(item.image.image_hash)

            self._save_state()

    def mark_failed(self, job_id: str, error_message: str):
        """
        Mark an item as failed.

        Args:
            job_id: Job ID
            error_message: Error description
        """
        if job_id in self.items:
            item = self.items[job_id]
            item.error_message = error_message
            item.worker_id = item.lease_until = None

            # Move to review if max retries exceeded
            if item.attempts >= settings.image_max_retries:
                item.status = "review"
                logger.error(
                    f"Max retries exceeded for {item.image.image_title}: {error_message}"
                )
            else:
                item.status = "failed"

            self._save_state()

    def retry_failed(self) -> int:
        """
        Reset failed items to pending for retry.

        Returns:
            Number of items reset
        """
        count = 0
        for item in self.items.values():
            if item.status == "failed":
                item.status = "pending"
                item.error_message = None
                count += 1

        if count > 0:
            self._save_state()
            logger.info(f"Reset {count} failed items for retry")

        return count

    def estimate_time_remaining(self) -> timedelta:
        """
        Estimate time remaining for all pending items.

        Uses average processing time per megapixel from completed items.

        Returns:
            Estimated timedelta
        """
        pending = [item for item in self.items.values() if item.status == "pending"]

        if not pending:
            return timedelta(0)

        # Calculate total megapixels remaining
        total_megapixels = sum(
            item.image.megapixels for item in pending if item.image.megapixels
        )

        # Use average time per megapixel from history
        if self.processing_times_per_mp:
            avg_time_per_mp = mean(self.processing_times_per_mp)
        else:
            # Default estimate: ~15 seconds per megapixel on CPU
            avg_time_per_mp = 15.0

        estimated_seconds = total_megapixels * avg_time_per_mp
        return timedelta(seconds=estimated_seconds)

    def get_status_summary(self) -> Dict:
        """
        Get queue status summary.

        Returns:
            Dict with counts by status and time estimate
        """
        by_status = {"pending": 0, "processing": 0, "completed": 0, "failed": 0, "review": 0}

        for item in self.items.values():
            by_status[item.status] = by_status.get(item.status, 0) + 1

        eta = self.estimate_time_remaining()
        eta_str = f"{int(eta.total_seconds() // 60)} minutes" if eta.total_seconds() > 0 else "0 minutes"

        return {
            "total": len(self.items),
            "by_status": by_status,
            "estimated_remaining": eta_str,
        }
