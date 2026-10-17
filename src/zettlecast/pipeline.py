"""
End-to-end processing shared by the CLI and the API.

    ingest_note()        note -> chunks + embeddings in LanceDB -> graph edges
    ingest_path()        file -> note (PDF/markdown/image) or podcast queue (audio)
    run_podcast_queue()  pending episodes -> transcript -> LLM enhancement ->
                         markdown note -> ingest_note()

GPU sharing: Ollama's model (~22 GB) and NeMo ASR (~14 GB) don't fit on the
card together, so the LLM is unloaded before each transcription and NeMo frees
its models before enhancement starts.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import settings
from .db import db
from .linker import update_edges_for_note
from .models import NoteModel, ProcessingResult
from .parser import AUDIO_EXTENSIONS, parse_file

logger = logging.getLogger(__name__)


def ingest_note(note: NoteModel) -> NoteModel:
    """Store a parsed note (replacing any previous version) and link it into the graph."""
    db.upsert_note(note)
    update_edges_for_note(note.uuid)
    return note


def ingest_path(path: Path, podcast_name: Optional[str] = None) -> ProcessingResult:
    """
    Ingest one file.

    Audio files are added to the podcast queue (status "queued"); run the
    queue to transcribe them. Everything else is parsed and stored now.
    """
    path = Path(path)
    if path.suffix.lower() in AUDIO_EXTENSIONS:
        from .podcast.queue import DuplicateEpisodeError, TranscriptionQueue

        try:
            job_id = TranscriptionQueue().add(path, podcast_name=podcast_name)
        except DuplicateEpisodeError:
            return ProcessingResult(status="duplicate", title=path.stem)
        return ProcessingResult(status="queued", uuid=job_id, title=path.stem)

    result = parse_file(path)
    if result.status == "success" and result.note:
        existing = db.get_note_by_hash(result.note.content_hash)
        if existing and existing.uuid != result.note.uuid:
            return ProcessingResult(status="duplicate", uuid=existing.uuid, title=existing.title)
        ingest_note(result.note)
    return result


# --- Podcast queue ---

# (stage, chunk, total_chunks, device) -> False to cancel
ProgressCallback = Callable[[str, int, int, str], bool]


@dataclass
class EpisodeOutcome:
    episode_id: str
    title: str
    status: str  # completed | failed | cancelled
    note_path: Optional[Path] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    processing_seconds: float = 0.0
    speakers: int = 0


@dataclass
class QueueRunSummary:
    outcomes: list = field(default_factory=list)

    @property
    def completed(self) -> int:
        return sum(o.status == "completed" for o in self.outcomes)

    @property
    def failed(self) -> int:
        return sum(o.status == "failed" for o in self.outcomes)


def known_speakers(show: str, exclude_uuid: str = "") -> list[str]:
    """
    Full names identified in earlier transcripts of the same show, most frequent
    first. Lets the LLM resolve "Coach Adam" to a name spelled out elsewhere.
    """
    from collections import Counter

    from .identity import parse_frontmatter

    if not show:
        return []
    counts: Counter = Counter()
    for path in settings.transcripts_path.glob("*.md"):
        try:
            meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if (meta.get("podcast") or {}).get("show") != show or meta.get("uuid") == exclude_uuid:
            continue
        counts.update({n for n in (meta.get("speaker_names") or {}).values() if n and " " in n.strip()})
    return [name for name, _ in counts.most_common(20)]


def run_podcast_queue(
    limit: Optional[int] = None,
    enhance: bool = True,
    sync: bool = True,
    on_episode_start: Optional[Callable[[str, str], None]] = None,
    on_episode_done: Optional[Callable[[EpisodeOutcome], None]] = None,
    progress_for: Optional[Callable[[str], ProgressCallback]] = None,
) -> QueueRunSummary:
    """
    Process pending podcast episodes end to end.

    Args:
        limit: Max episodes to process (None = all pending)
        enhance: Run LLM analysis + cleanup
        sync: First add any audio under storage/podcasts that isn't queued yet
        on_episode_start: Called with (episode_id, title) before each episode
        on_episode_done: Called with the outcome of each episode
        progress_for: Returns a progress callback for an episode id; the
            callback returning False cancels that episode
    """
    from . import llm
    from .podcast.enhancer import TranscriptEnhancer
    from .podcast.formatter import save_result
    from .podcast.nemo_transcriber import CancellationError, NeMoTranscriber
    from .podcast.queue import TranscriptionQueue

    queue = TranscriptionQueue()
    if sync:
        queue.sync_with_storage()

    transcriber = NeMoTranscriber()
    enhancer = TranscriptEnhancer() if enhance else None
    summary = QueueRunSummary()

    while limit is None or len(summary.outcomes) < limit:
        item = queue.get_next_pending()
        if not item:
            break
        episode = item.episode
        title = episode.episode_title or Path(episode.audio_path).stem
        if on_episode_start:
            on_episode_start(episode.id, title)
        progress = progress_for(episode.id) if progress_for else None

        def check_cancel(stage: str) -> None:
            if progress and progress(stage, 0, 0, "cuda") is False:
                raise CancellationError("Cancelled by user")

        queue.mark_started(episode.id)
        started = time.time()
        outcome = EpisodeOutcome(episode_id=episode.id, title=title, status="failed")
        try:
            llm.unload()  # make room on the GPU for ASR
            result = transcriber.transcribe(Path(episode.audio_path), episode=episode, progress_callback=progress)

            enhanced = None
            if enhancer:
                check_cancel("enhancing")
                enhanced = enhancer.enhance(
                    result.full_text,
                    podcast=episode.podcast_name or "",
                    episode=title,
                    show_notes=episode.description or "",
                    known_speakers=known_speakers(episode.podcast_name or "", exclude_uuid=episode.id),
                )
                result.keywords = enhanced["keywords"]
                result.sections = enhanced["sections"]
                result.summary = enhanced["summary"]
                result.key_points = enhanced["key_points"]

            check_cancel("saving")
            note_path = save_result(result, episode, enhanced)
            parsed = parse_file(note_path)
            if parsed.status != "success" or not parsed.note:
                raise RuntimeError(f"Could not read back transcript note: {parsed.error_message}")
            check_cancel("indexing")
            ingest_note(parsed.note)

            queue.mark_completed(episode.id, result, note_path)
            outcome = EpisodeOutcome(
                episode_id=episode.id,
                title=title,
                status="completed",
                note_path=note_path,
                duration_seconds=result.duration_seconds,
                processing_seconds=time.time() - started,
                speakers=result.speakers_detected,
            )
        except CancellationError:
            queue.mark_cancelled(episode.id)
            outcome.status = "cancelled"
        except Exception as e:
            logger.exception(f"Episode failed: {title}")
            queue.mark_failed(episode.id, str(e), max_retries=settings.podcast_max_retries)
            outcome.error = str(e)

        summary.outcomes.append(outcome)
        if on_episode_done:
            on_episode_done(outcome)

    return summary


def reindex_all() -> int:
    """Re-parse every transcript/markdown note on disk and rebuild the index and graph."""
    from .linker import rebuild_graph

    count = 0
    for path in sorted(settings.transcripts_path.glob("*.md")):
        result = parse_file(path)
        if result.status == "success" and result.note:
            db.upsert_note(result.note)
            count += 1
    rebuild_graph()
    return count
