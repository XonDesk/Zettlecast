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
import sys
import time
from contextlib import contextmanager
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


@contextmanager
def keep_awake(reason: str = "Zettlecast is transcribing podcasts"):
    """
    Hold a Windows power request so idle sleep (including Modern Standby) doesn't
    pause a long run. Released when the block exits; no-op elsewhere or on failure.
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes
    from ctypes import wintypes

    class ReasonContext(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("Flags", wintypes.DWORD), ("Reason", wintypes.LPWSTR)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.PowerCreateRequest.restype = wintypes.HANDLE
    kernel32.PowerCreateRequest.argtypes = [ctypes.POINTER(ReasonContext)]
    kernel32.PowerSetRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
    kernel32.PowerClearRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    SYSTEM_REQUIRED, EXECUTION_REQUIRED = 1, 3  # POWER_REQUEST_TYPE

    context = ReasonContext(0, 1, reason)  # POWER_REQUEST_CONTEXT_SIMPLE_STRING
    handle = kernel32.PowerCreateRequest(ctypes.byref(context))
    if not handle or handle == wintypes.HANDLE(-1).value:
        logger.warning("Could not create a power request; the PC may sleep during processing")
        yield
        return
    held = [t for t in (SYSTEM_REQUIRED, EXECUTION_REQUIRED) if kernel32.PowerSetRequest(handle, t)]
    try:
        yield
    finally:
        for t in held:
            kernel32.PowerClearRequest(handle, t)
        kernel32.CloseHandle(handle)


def known_speakers(show: str, exclude_uuid: str = "") -> list[str]:
    """
    Full names identified in earlier transcripts of the same show: names the
    publisher confirmed (verified_speakers) first, then the rest by frequency,
    leaving out near-duplicates of confirmed names ("Ryan" vs "Ryne Anderson").
    Lets the LLM resolve "Coach Adam" to a name spelled out elsewhere.
    """
    import difflib
    from collections import Counter

    from .identity import parse_frontmatter

    if not show:
        return []
    counts: Counter = Counter()
    verified: Counter = Counter()
    for path in settings.transcripts_path.glob("*.md"):
        try:
            meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if (meta.get("podcast") or {}).get("show") != show or meta.get("uuid") == exclude_uuid:
            continue
        counts.update({n for n in (meta.get("speaker_names") or {}).values() if n and " " in n.strip()})
        verified.update(set(meta.get("verified_speakers") or []))
    confirmed = [name for name, _ in verified.most_common()]
    others = [
        name for name, _ in counts.most_common()
        if name not in verified and not difflib.get_close_matches(name, confirmed, n=1, cutoff=0.85)
    ]
    return (confirmed + others)[:20]


def episode_reference(episode) -> list:
    """Words of the transcript the publisher put in the feed ([] if none or unreadable)."""
    from .podcast.reference import cache_path_for, load_reference

    if not episode.transcript_url:
        return []
    try:
        return load_reference(
            episode.transcript_url, cache_path_for(Path(episode.audio_path), episode.transcript_url)
        )
    except Exception as e:
        logger.warning(f"Could not load published transcript: {e}")
        return []


def published_speaker_names(ref: list, result) -> dict:
    """
    Speaker label -> name from the publisher's transcript, when it names its speakers.
    """
    from .podcast.reference import speaker_names_from_reference

    if not ref:
        return {}
    turns = [(s.start, s.end, s.speaker, s.text) for s in result.segments if s.speaker]
    names = speaker_names_from_reference(turns, ref)
    if names:
        logger.info(f"Speaker names from the published transcript: {names}")
    return names


_scorer = None
_lookup = None


def acoustic_scorer():
    """Parakeet as an acoustic judge (~1.3 GB, kept loaded; fits next to the LLM)."""
    global _scorer
    if _scorer is None:
        import nemo.collections.asr as nemo_asr

        from .podcast.acoustic import AcousticScorer
        from .podcast.nemo_transcriber import NeMoTranscriber

        NeMoTranscriber._quiet_nemo()
        model = nemo_asr.models.ASRModel.from_pretrained(settings.nemo_parakeet_model)
        _scorer = AcousticScorer(model.cuda().eval().half())
    return _scorer


def name_lookup():
    """Online name lookup for the verifier (None when NAME_LOOKUP is off); queries are cached on disk."""
    global _lookup
    if not settings.name_lookup:
        return None
    if _lookup is None:
        from .podcast.lookup import NameLookup

        _lookup = NameLookup(settings.storage_path / "lookups" / "wikipedia.json")
    _lookup.enabled = True  # retry after an earlier network error
    return _lookup


def _run_podcast_queue(
    limit: Optional[int] = None,
    enhance: bool = True,
    verify: bool = True,
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
        verify: Check corrections against the audio (needs enhance)
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
            verifier = None
            if enhancer:
                check_cancel("enhancing")
                ref = episode_reference(episode)
                speakers = known_speakers(episode.podcast_name or "", exclude_uuid=episode.id)
                if verify:
                    from .podcast.verifier import TranscriptVerifier, glossary_path, load_audio, load_glossary

                    glossary_file = glossary_path(settings.storage_path, episode.podcast_name or "")
                    verifier = TranscriptVerifier(
                        result.segments, load_audio(Path(episode.audio_path)), acoustic_scorer(),
                        reference=ref, glossary=load_glossary(glossary_file), lookup=name_lookup(),
                    )
                enhanced = enhancer.enhance(
                    result.full_text,
                    podcast=episode.podcast_name or "",
                    episode=title,
                    show_notes=episode.description or "",
                    known_speakers=speakers,
                    published_speakers=published_speaker_names(ref, result),
                    verifier=verifier,
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
            if verifier is not None:
                from .podcast.verifier import save_glossary

                save_glossary(glossary_file, verifier.learn(
                    list(enhanced["speakers"].values()), verified=enhanced.get("verified_speakers", [])
                ))
                if verifier.lookup is not None:
                    verifier.lookup.save()

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


def run_podcast_queue(*args, **kwargs) -> QueueRunSummary:
    """
    Process pending podcast episodes end to end, keeping the PC awake meanwhile.
    Arguments: see _run_podcast_queue.
    """
    from . import embeddings

    with keep_awake(), embeddings.hold():
        return _run_podcast_queue(*args, **kwargs)


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
