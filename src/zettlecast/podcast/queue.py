"""
Transcription Queue Manager

Persistent queue for batch podcast processing with retry logic
and time estimation.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from ..config import settings
from .models import PodcastEpisode, QueueItem, TranscriptionResult

logger = logging.getLogger(__name__)


class DuplicateEpisodeError(Exception):
    """Raised when attempting to add a duplicate episode."""
    pass


class TranscriptionQueue:
    """
    Persistent queue for batch podcast transcription.

    State is saved to JSON file for crash recovery.
    Tracks processing times for accurate ETA estimation.
    """

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or (settings.storage_path / "podcast_queue.json")
        self.items: Dict[str, QueueItem] = {}
        self.processing_times: List[float] = []
        self.completed_hashes: List[str] = []

        self._load_state()

    def _load_state(self):
        """Load queue state from disk."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))

                for job_id, item_data in data.get("items", {}).items():
                    # Parse nested episode
                    episode_data = item_data.pop("episode", {})
                    episode = PodcastEpisode(**episode_data)
                    
                    # Parse datetime fields
                    for dt_field in ["added_at", "started_at", "completed_at"]:
                        if item_data.get(dt_field):
                            item_data[dt_field] = datetime.fromisoformat(item_data[dt_field])
                    
                    self.items[job_id] = QueueItem(episode=episode, **item_data)

                self.processing_times = data.get("processing_times", [])
                self.completed_hashes = data.get("completed_hashes", [])

                logger.info(f"Loaded queue state: {len(self.items)} items")

            except Exception as e:
                logger.error(f"Failed to load queue state: {e}")
                self.items = {}

    def _save_state(self):
        """Save queue state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "items": {},
                "processing_times": self.processing_times[-100:],  # Keep last 100
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
    def compute_audio_hash(audio_path: Path) -> str:
        """Compute SHA256 hash of audio file."""
        sha256 = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def add(
        self,
        audio_path: Path,
        podcast_name: Optional[str] = None,
        episode_title: Optional[str] = None,
        feed_url: Optional[str] = None,
        skip_duplicate: bool = True,
    ) -> str:
        """
        Add an episode to the queue.

        Args:
            audio_path: Path to audio file
            podcast_name: Name of the podcast/show
            episode_title: Episode title
            feed_url: RSS feed URL
            skip_duplicate: Whether to skip if already processed

        Returns:
            Job ID

        Raises:
            DuplicateEpisodeError: If episode already processed and skip_duplicate=True
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Compute hash for deduplication
        audio_hash = self.compute_audio_hash(audio_path)

        # Check for duplicate
        if skip_duplicate and audio_hash in self.completed_hashes:
            raise DuplicateEpisodeError(
                f"Episode already processed: {audio_path.name} (hash: {audio_hash[:8]})"
            )

        # Check if already in queue
        for item in self.items.values():
            if item.episode.audio_hash == audio_hash:
                logger.info(f"Episode already in queue: {audio_path.name}")
                return item.episode.id

        # Create episode
        episode = PodcastEpisode(
            audio_path=str(audio_path.absolute()),
            audio_hash=audio_hash,
            podcast_name=podcast_name,
            episode_title=episode_title or audio_path.stem,
            feed_url=feed_url,
        )

        # Create queue item
        item = QueueItem(
            episode=episode,
            status="pending",
            added_at=datetime.utcnow(),
        )

        self.items[episode.id] = item
        self._save_state()

        logger.info(f"Added to queue: {episode.episode_title} (ID: {episode.id[:8]})")
        return episode.id

    def add_directory(
        self,
        directory: Path,
        podcast_name: Optional[str] = None,
        recursive: bool = True,
        extensions: List[str] = None,
    ) -> List[str]:
        """
        Add all audio files from a directory.

        Args:
            directory: Directory to scan
            podcast_name: Name to assign to all episodes
            recursive: Whether to scan subdirectories
            extensions: File extensions to include

        Returns:
            List of added job IDs
        """
        extensions = extensions or ["mp3", "wav", "m4a", "aac", "ogg", "opus", "flac"]
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        job_ids = []
        pattern = "**/*" if recursive else "*"

        for ext in extensions:
            for audio_file in directory.glob(f"{pattern}.{ext}"):
                try:
                    job_id = self.add(
                        audio_path=audio_file,
                        podcast_name=podcast_name or directory.name,
                    )
                    job_ids.append(job_id)
                except DuplicateEpisodeError:
                    logger.debug(f"Skipping duplicate: {audio_file.name}")
                except Exception as e:
                    logger.error(f"Failed to add {audio_file.name}: {e}")

        logger.info(f"Added {len(job_ids)} episodes from {directory}")
        return job_ids

    def add_from_feed(
        self,
        feed_url: str,
        limit: int = 5,
        download_dir: Optional[Path] = None,
    ) -> List[str]:
        """
        Add episodes from an RSS feed.

        Args:
            feed_url: RSS feed URL
            limit: Max episodes to add
            download_dir: Directory to store downloaded audio

        Returns:
            List of added job IDs
        """
        from .rss import parse_feed, download_episode, sanitize_filename

        # Default download dir
        if not download_dir:
            download_dir = settings.storage_path / "podcasts"

        feed = parse_feed(feed_url, limit=limit)
        job_ids = []
        
        # Create podcast subdirectory
        podcast_dir = download_dir / sanitize_filename(feed.title)
        
        logger.info(f"Processing feed: {feed.title} ({len(feed.episodes)} episodes)")

        for episode in feed.episodes:
            try:
                # Check if we should download
                # We can't easily check hash before download, but we can check if file exists
                # or if we have a completed job with same title/feed
                
                # Download
                audio_path = download_episode(episode, podcast_dir)
                
                # Add to queue
                job_id = self.add(
                    audio_path=audio_path,
                    podcast_name=feed.title,
                    episode_title=episode.title,
                    feed_url=feed_url,
                )
                job_ids.append(job_id)
                
            except DuplicateEpisodeError:
                logger.debug(f"Skipping duplicate: {episode.title}")
            except Exception as e:
                logger.error(f"Failed to process episode {episode.title}: {e}")

        logger.info(f"Added {len(job_ids)} episodes from feed")
        return job_ids

    def get_next_pending(self) -> Optional[QueueItem]:
        """Get the next pending item to process."""
        for item in self.items.values():
            if item.status == "pending":
                return item
        return None

    def get_pending_count(self) -> int:
        """Get number of pending items."""
        return sum(1 for i in self.items.values() if i.status == "pending")

    def get_failed_count(self) -> int:
        """Get number of failed items marked for review."""
        return sum(1 for i in self.items.values() if i.status == "review")

    def estimate_time_remaining(self) -> timedelta:
        """
        Estimate time to complete all pending items.

        Uses average processing time from completed items,
        or assumes 6 min/hr audio as default.
        """
        pending = self.get_pending_count()
        if pending == 0:
            return timedelta(0)

        if self.processing_times:
            avg_time = mean(self.processing_times)
        else:
            # Default: ~6 min per episode (assuming 1-hour episodes)
            avg_time = 360

        return timedelta(seconds=avg_time * pending)

    def mark_started(self, job_id: str):
        """Mark a job as started."""
        if job_id in self.items:
            self.items[job_id].status = "processing"
            self.items[job_id].started_at = datetime.utcnow()
            self.items[job_id].attempts += 1
            self._save_state()

    def mark_completed(self, job_id: str, result: TranscriptionResult, result_path: Path):
        """Mark a job as completed."""
        if job_id in self.items:
            item = self.items[job_id]
            item.status = "completed"
            item.completed_at = datetime.utcnow()
            item.result_path = str(result_path)

            # Track processing time
            self.processing_times.append(result.processing_time_seconds)
            self.completed_hashes.append(item.episode.audio_hash)

            self._save_state()
            logger.info(f"Completed: {item.episode.episode_title}")

    def mark_failed(self, job_id: str, error: str, max_retries: int = 3):
        """
        Mark a job as failed.

        If attempts < max_retries, keeps as pending for retry.
        Otherwise marks as 'review' for manual intervention.
        """
        if job_id in self.items:
            item = self.items[job_id]
            item.error_message = error

            if item.attempts >= max_retries:
                item.status = "review"
                logger.warning(
                    f"Marked for review after {item.attempts} attempts: "
                    f"{item.episode.episode_title}"
                )
            else:
                item.status = "pending"  # Will retry
                logger.info(
                    f"Will retry ({item.attempts}/{max_retries}): "
                    f"{item.episode.episode_title}"
                )

            self._save_state()

    def retry_failed(self):
        """Reset all 'review' items to 'pending' for retry."""
        count = 0
        for item in self.items.values():
            if item.status == "review":
                item.status = "pending"
                item.attempts = 0
                item.error_message = None
                count += 1

        if count > 0:
            self._save_state()
            logger.info(f"Reset {count} failed items for retry")

        return count

    def get_status_summary(self) -> dict:
        """Get summary of queue status."""
        status_counts = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "review": 0,
        }

        for item in self.items.values():
            status_counts[item.status] = status_counts.get(item.status, 0) + 1

        eta = self.estimate_time_remaining()

        return {
            "total": len(self.items),
            "by_status": status_counts,
            "estimated_remaining": str(eta),
            "avg_processing_time": (
                f"{mean(self.processing_times):.1f}s"
                if self.processing_times
                else "unknown"
            ),
        }
