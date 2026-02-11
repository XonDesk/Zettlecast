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

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or (settings.storage_path / "image_queue.json")
        self.items: Dict[str, QueueItem] = {}
        self.processing_times_per_mp: List[float] = []  # Seconds per megapixel
        self.completed_hashes: List[str] = []

        self._load_state()

    def _load_state(self):
        """Load queue state from disk."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))

                for job_id, item_data in data.get("items", {}).items():
                    # Parse nested image
                    image_data = item_data.pop("image", {})
                    image = ImageItem(**image_data)

                    # Parse datetime fields
                    for dt_field in ["added_at", "started_at", "completed_at"]:
                        if item_data.get(dt_field):
                            item_data[dt_field] = datetime.fromisoformat(item_data[dt_field])

                    self.items[job_id] = QueueItem(image=image, **item_data)

                self.processing_times_per_mp = data.get("processing_times_per_mp", [])
                self.completed_hashes = data.get("completed_hashes", [])

                logger.info(f"Loaded queue state: {len(self.items)} items")

                # Auto-reset stuck 'processing' items
                self._reset_stuck_items()

            except Exception as e:
                logger.error(f"Failed to load queue state: {e}")
                self.items = {}

    def _reset_stuck_items(self):
        """
        Reset items stuck in 'processing' state.

        Items that have been 'processing' for more than 1 hour are considered stuck
        (e.g., due to process crash).
        """
        now = datetime.utcnow()
        stuck_threshold = timedelta(hours=1)
        reset_count = 0

        for item in self.items.values():
            if item.status == "processing":
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

    def _save_state(self):
        """Save queue state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "items": {},
                "processing_times_per_mp": self.processing_times_per_mp[-100:],  # Keep last 100
                "completed_hashes": self.completed_hashes[-1000:],  # Keep last 1000
            }

            for job_id, item in self.items.items():
                item_dict = item.model_dump()
                # Convert datetime to ISO format
                for dt_field in ["added_at", "started_at", "completed_at"]:
                    if item_dict.get(dt_field):
                        item_dict[dt_field] = item_dict[dt_field].isoformat()
                data["items"][job_id] = item_dict

            self.state_file.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )

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
