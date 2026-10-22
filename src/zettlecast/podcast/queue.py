"""
Transcription Queue Manager

Persistent queue for batch podcast processing with retry logic
and time estimation.
"""

import hashlib
import json
import logging
import os
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from .. import paths, statefile
from ..config import settings
from .models import PodcastEpisode, QueueItem, TranscriptionResult

logger = logging.getLogger(__name__)


class DuplicateEpisodeError(Exception):
    """Raised when attempting to add a duplicate episode."""
    pass


def _worker_alive(pid: Optional[int], started_at: Optional[datetime]) -> Optional[bool]:
    """
    Whether the process that started an item still runs: True / False, or None when
    unknown (no pid recorded, or psutil unavailable). A process started after the item
    reuses the pid of one that exited.
    """
    if not pid:
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        proc = psutil.Process(pid)
        if started_at and datetime.utcfromtimestamp(proc.create_time()) > started_at:
            return False
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return None


class TranscriptionQueue:
    """
    Persistent queue for batch podcast transcription.

    State is saved to JSON file for crash recovery.
    Tracks processing times for accurate ETA estimation.
    """

    DATETIME_FIELDS = ("added_at", "started_at", "completed_at", "lease_until")
    LISTS = {"processing_times": 100, "completed_hashes": 1000}  # entries kept

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or (settings.storage_path / "podcast_queue.json")
        self.items: Dict[str, QueueItem] = {}
        self.processing_times: List[float] = []
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
            episode = PodcastEpisode(**item_data.pop("episode", {}))
            for dt_field in self.DATETIME_FIELDS:
                if item_data.get(dt_field):
                    item_data[dt_field] = datetime.fromisoformat(item_data[dt_field])
            self.items[job_id] = QueueItem(episode=episode, **item_data)
        self.processing_times = list(data.get("processing_times", []))
        self.completed_hashes = list(data.get("completed_hashes", []))

    def _reset_stuck_items(self):
        """
        Reset items stuck in 'processing' state.

        This can happen when the process crashes or is killed during transcription.
        An item claimed by a worker (another machine) is stuck once its lease runs out; one
        started by a process here once that process has exited; for items without a recorded
        process, after 1 hour of 'processing'.
        """
        now = datetime.utcnow()
        stuck_threshold = timedelta(hours=1)
        reset_count = 0

        for item in self.items.values():
            if item.status == "processing" and item.lease_until:
                if item.lease_until < now:
                    logger.warning(f"Lease ran out, back to pending: {item.episode.episode_title}")
                    item.status = "pending"
                    item.started_at = item.worker_id = item.lease_until = None
                    reset_count += 1
            elif item.status == "processing":
                alive = _worker_alive(item.worker_pid, item.started_at)
                if alive is False:
                    logger.warning(f"Resetting job whose process exited: {item.episode.episode_title}")
                    item.status = "pending"
                    item.started_at = item.worker_pid = None
                    reset_count += 1
                elif alive is None and item.started_at:
                    processing_time = now - item.started_at
                    if processing_time > stuck_threshold:
                        logger.warning(
                            f"Resetting stuck job (processing for {processing_time}): "
                            f"{item.episode.episode_title}"
                        )
                        item.status = "pending"
                        item.started_at = None
                        reset_count += 1
                elif alive is None:
                    # No start time but marked as processing - reset it
                    item.status = "pending"
                    reset_count += 1
        
        if reset_count > 0:
            logger.info(f"Auto-reset {reset_count} stuck 'processing' items to 'pending'")
            self._save_state()

    def _serialize(self) -> dict:
        data = {
            "items": {},
            "processing_times": list(self.processing_times),
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

    def reload(self) -> None:
        """Read the file again (changes other writers made)."""
        self.items = {}
        self._load_state()

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
        description: Optional[str] = None,
        published: Optional[str] = None,
        transcript_url: Optional[str] = None,
        guid: Optional[str] = None,
        skip_duplicate: bool = True,
    ) -> str:
        """
        Add an episode to the queue.

        If the audio is already queued or processed, any feed details it lacks
        (feed URL, show notes, publish date, transcript link) are filled in on the existing item.

        Args:
            audio_path: Path to audio file
            podcast_name: Name of the podcast/show
            episode_title: Episode title
            feed_url: RSS feed URL
            description: Show notes from the feed
            published: Publish date from the feed
            transcript_url: Transcript published with the episode
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

        existing = next((i for i in self.items.values() if i.episode.audio_hash == audio_hash), None)
        if existing:
            self._fill_feed_details(existing, feed_url=feed_url, description=description,
                                    published=published, transcript_url=transcript_url, guid=guid)

        if skip_duplicate and (
            audio_hash in self.completed_hashes or (existing and existing.status == "completed")
        ):
            raise DuplicateEpisodeError(
                f"Episode already processed: {audio_path.name} (hash: {audio_hash[:8]})"
            )

        if existing:
            logger.info(f"Episode already in queue: {audio_path.name}")
            return existing.episode.id

        # Create episode
        episode = PodcastEpisode(
            audio_path=str(audio_path.absolute()),
            audio_hash=audio_hash,
            podcast_name=podcast_name,
            episode_title=episode_title or audio_path.stem,
            feed_url=feed_url,
            description=description,
            published=published,
            transcript_url=transcript_url,
            guid=guid,
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

    def _fill_feed_details(self, item: QueueItem, **feed_info):
        """Set the feed details an item lacks (feed URL, show notes, publish date, transcript link, guid)."""
        missing = {k: v for k, v in feed_info.items() if v and not getattr(item.episode, k)}
        if missing:
            for k, v in missing.items():
                setattr(item.episode, k, v)
            self._save_state()
            logger.info(f"Added feed details to {item.episode.episode_title}: {', '.join(missing)}")

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

    def find_feed_episode(
        self, feed_url: str, episode, podcast_dir: Path, filename: Optional[str] = None
    ) -> Optional[QueueItem]:
        """
        The queue item for a feed episode, if any: same guid, else (items imported before guids
        were kept) same feed and title, or same downloaded file. Failed and cancelled items don't count.
        """
        from .rss import sanitize_filename

        filename = filename or sanitize_filename(episode.title)
        for item in self.items.values():
            if item.status in ("failed", "cancelled"):
                continue
            ep = item.episode
            if episode.guid and ep.guid:
                if ep.guid == episode.guid:
                    return item
                continue
            if (ep.feed_url == feed_url and ep.episode_title == episode.title
                    and (ep.published or "") == (episode.published or "")):
                return item
            path = Path(paths.strip(ep.audio_path))
            if path.stem == filename and (
                path.parent.name == podcast_dir.name if paths.is_worker_ref(ep.audio_path)
                else path.parent.resolve() == podcast_dir.resolve()
            ):
                return item
        return None

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
        from .rss import download_episode, episode_date, parse_feed, sanitize_filename

        # Default download dir
        if not download_dir:
            download_dir = settings.podcasts_path

        feed = parse_feed(feed_url, limit=limit)
        job_ids = []
        
        # Create podcast subdirectory
        podcast_dir = download_dir / sanitize_filename(feed.title)
        
        logger.info(f"Processing feed: {feed.title} ({len(feed.episodes)} episodes)")

        # Episodes sharing a title ("Summer Q&A") get the publish date in their file name
        title_counts = Counter(sanitize_filename(e.title) for e in feed.episodes)

        for episode in feed.episodes:
            stem = sanitize_filename(episode.title)
            if title_counts[stem] > 1:
                stem = f"{stem} {episode_date(episode.published)}"
            known = self.find_feed_episode(feed_url, episode, podcast_dir, stem)
            if known:
                # Same feed item: no download, even if it was retitled or its audio re-encoded
                self._fill_feed_details(known, feed_url=feed_url, description=episode.description,
                                        published=episode.published, transcript_url=episode.transcript_url,
                                        guid=episode.guid)
                logger.debug(f"Already known: {episode.title}")
                continue
            try:
                audio_path = download_episode(episode, podcast_dir, stem=stem)
                
                # Add to queue
                job_id = self.add(
                    audio_path=audio_path,
                    podcast_name=feed.title,
                    episode_title=episode.title,
                    feed_url=feed_url,
                    description=episode.description,
                    published=episode.published,
                    transcript_url=episode.transcript_url,
                    guid=episode.guid,
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

    def mark_started(self, job_id: str, remote: bool = False):
        """Mark a job as started (remote: by a worker on another machine, which holds a lease)."""
        if job_id in self.items:
            self.items[job_id].status = "processing"
            self.items[job_id].started_at = datetime.utcnow()
            self.items[job_id].worker_pid = None if remote else os.getpid()
            self.items[job_id].attempts += 1
            self._save_state()

    def set_path(self, job_id: str, path: str) -> None:
        """Point a job at another copy of its file (the worker's, once it has fetched it)."""
        if job_id in self.items:
            self.items[job_id].episode.audio_path = path
            self._save_state()

    def claim(self, worker_id: str, lease: timedelta, limit: Optional[int] = None) -> List[QueueItem]:
        """Hand pending jobs to a worker: processing under a lease it renews with each heartbeat."""
        pending = [i for i in self.items.values() if i.status == "pending"]
        if limit is not None:
            pending = pending[:limit]
        now = datetime.utcnow()
        for item in pending:
            item.status = "processing"
            item.started_at = now
            item.worker_id = worker_id
            item.worker_pid = None
            item.lease_until = now + lease
        if pending:
            self._save_state()
        return [self.items[i.episode.id] for i in pending]

    def renew(self, worker_id: str, lease: timedelta) -> int:
        """Extend the leases of the jobs a worker holds."""
        held = [i for i in self.items.values() if i.status == "processing" and i.worker_id == worker_id]
        for item in held:
            item.lease_until = datetime.utcnow() + lease
        if held:
            self._save_state()
        return len(held)

    @staticmethod
    def _unclaim(item: QueueItem) -> None:
        item.worker_pid = item.worker_id = item.lease_until = None

    def release(self, job_ids) -> int:
        """Put unfinished jobs of a stopped run back to pending (the attempt isn't counted)."""
        count = 0
        for job_id in job_ids:
            item = self.items.get(job_id)
            if item and item.status == "processing":
                item.status = "pending"
                item.started_at = None
                self._unclaim(item)
                item.attempts = max(0, item.attempts - 1)
                count += 1
        if count:
            self._save_state()
        return count

    def mark_completed(self, job_id: str, result: TranscriptionResult, result_path: Path):
        """Mark a job as completed."""
        if job_id in self.items:
            item = self.items[job_id]
            item.status = "completed"
            item.completed_at = datetime.utcnow()
            item.result_path = str(result_path)
            self._unclaim(item)

            # Track processing time
            self.processing_times.append(result.processing_time_seconds)
            if item.episode.audio_hash not in self.completed_hashes:
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
            self._unclaim(item)

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

    def reprocess(self, match: Optional[str] = None) -> int:
        """
        Put completed episodes back to pending so they are transcribed again.
        The note keeps its UUID (the episode id), so it is replaced in place.

        Args:
            match: Only episodes whose title contains this text (case-insensitive);
                None means every completed episode
        """
        count = 0
        for item in self.items.values():
            title = (item.episode.episode_title or "").lower()
            if item.status == "completed" and (not match or match.lower() in title):
                item.status = "pending"
                item.attempts = 0
                item.error_message = None
                count += 1
        if count:
            self._save_state()
        return count

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
    
    def reset_all_stuck(self) -> int:
        """
        Reset all stuck items (processing or review) to pending.
        
        Use this to recover from crashed/killed transcription runs.
        
        Returns:
            Number of items reset
        """
        count = 0
        for item in self.items.values():
            if item.status in ("processing", "review"):
                item.status = "pending"
                item.attempts = 0
                item.started_at = None
                item.error_message = None
                count += 1
        
        if count > 0:
            self._save_state()
            logger.info(f"Reset {count} stuck/failed items to pending")
        
        return count

    def mark_cancelled(self, job_id: str):
        """Mark a job as cancelled. It will not be retried."""
        if job_id in self.items:
            item = self.items[job_id]
            item.status = "cancelled"
            item.error_message = "Cancelled by user"
            self._unclaim(item)
            self._save_state()
            logger.info(f"Cancelled: {item.episode.episode_title}")

    def get_status_summary(self) -> dict:
        """Get summary of queue status."""
        status_counts = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "review": 0,
            "cancelled": 0,
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
    
    def sync_with_storage(self, podcasts_dir: Optional[Path] = None, transcripts_dir: Optional[Path] = None) -> dict:
        """
        Sync queue state with actual files on disk.
        
        Scans downloaded podcast files and compares against existing transcripts
        to determine what still needs processing. Automatically:
        - Resets stuck 'processing' items to 'pending'
        - Adds downloaded podcasts that have no transcript
        - Marks items as completed if transcript exists
        
        Args:
            podcasts_dir: Directory containing downloaded podcasts
            transcripts_dir: Directory containing transcript markdown files
            
        Returns:
            Dict with sync statistics
        """
        podcasts_dir = podcasts_dir or settings.podcasts_path
        transcripts_dir = transcripts_dir or settings.transcripts_path
        
        stats = {
            "podcasts_found": 0,
            "transcripts_found": 0,
            "added_to_queue": 0,
            "reset_stuck": 0,
            "already_complete": 0,
        }
        
        # Step 1: Reset all stuck 'processing' items
        for item in self.items.values():
            if item.status == "processing":
                item.status = "pending"
                item.started_at = None
                stats["reset_stuck"] += 1
        
        # Step 2: Find all downloaded podcast files
        audio_extensions = [".mp3", ".m4a", ".wav", ".aac", ".ogg", ".opus", ".flac"]
        podcast_files = []
        
        if podcasts_dir.exists():
            for ext in audio_extensions:
                podcast_files.extend(podcasts_dir.rglob(f"*{ext}"))
        
        stats["podcasts_found"] = len(podcast_files)
        
        # Step 3: Find all transcript markdown files
        transcript_files = []
        if transcripts_dir.exists():
            transcript_files = list(transcripts_dir.glob("*.md"))
        
        stats["transcripts_found"] = len(transcript_files)
        
        # Build a set of transcript basenames (without UUID suffix) for matching
        # Transcript format: {title}_{uuid8}.md
        transcript_names = set()
        for tf in transcript_files:
            # Remove the _uuid8.md suffix to get title
            stem = tf.stem
            if len(stem) > 9 and stem[-9] == "_":
                # Has UUID suffix, remove it
                title_part = stem[:-9].lower().replace("_", " ")
            else:
                title_part = stem.lower().replace("_", " ")
            transcript_names.add(title_part)
        
        # Step 4: Check each podcast file
        for podcast_file in podcast_files:
            # Get the podcast title from filename
            title = podcast_file.stem.lower()
            
            # Check if already in queue
            already_in_queue = False
            for item in self.items.values():
                if paths.is_worker_ref(item.episode.audio_path):
                    continue  # held by the worker, not in this folder
                if Path(item.episode.audio_path).resolve() == podcast_file.resolve():
                    already_in_queue = True
                    # If completed or has transcript, skip
                    if item.status == "completed":
                        stats["already_complete"] += 1
                    break
            
            if already_in_queue:
                continue
            
            # Check if transcript exists (fuzzy match on title)
            has_transcript = any(
                title[:20] in t_name or t_name[:20] in title 
                for t_name in transcript_names
            )
            
            if has_transcript:
                stats["already_complete"] += 1
                continue
            
            # Add to queue
            try:
                # Try to determine podcast name from parent directory
                podcast_name = podcast_file.parent.name
                if podcast_name == "podcasts":
                    podcast_name = None
                    
                self.add(
                    audio_path=podcast_file,
                    podcast_name=podcast_name,
                    episode_title=podcast_file.stem,
                    skip_duplicate=True,
                )
                stats["added_to_queue"] += 1
            except DuplicateEpisodeError:
                stats["already_complete"] += 1
            except Exception as e:
                logger.warning(f"Failed to add {podcast_file.name}: {e}")
        
        # Save state if changes were made
        if stats["reset_stuck"] > 0 or stats["added_to_queue"] > 0:
            self._save_state()
        
        return stats

