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
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import settings
from .db import db
from .linker import update_edges_for_note
from .models import NoteModel, ProcessingResult
from .parser import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, parse_file
from .podcast.models import PodcastEpisode, TranscriptionResult

logger = logging.getLogger(__name__)


def ingest_note(note: NoteModel) -> NoteModel:
    """
    Store a parsed note (replacing any previous version) and link it into the graph. On the
    primary it waits for the worker's index, and the worker's batch links it.
    """
    if db.upsert_note(note):
        update_edges_for_note(note.uuid)
    return note


def ingest_path(path: Path, podcast_name: Optional[str] = None) -> ProcessingResult:
    """
    Ingest one file.

    Audio files are added to the podcast queue (status "queued"); run the
    queue to transcribe them. On the primary (no GPU) images go to the image
    queue the same way. Everything else is parsed and stored now.
    """
    path = Path(path)
    if path.suffix.lower() in AUDIO_EXTENSIONS:
        from .podcast.queue import DuplicateEpisodeError, TranscriptionQueue

        try:
            job_id = TranscriptionQueue().add(path, podcast_name=podcast_name)
        except DuplicateEpisodeError:
            return ProcessingResult(status="duplicate", title=path.stem)
        return ProcessingResult(status="queued", uuid=job_id, title=path.stem)
    if settings.is_primary and path.suffix.lower() in IMAGE_EXTENSIONS:
        from .image.queue import DuplicateImageError, ImageQueue

        try:
            job_id = ImageQueue().add(path, collection_name=podcast_name)
        except DuplicateImageError:
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
    """Parakeet as an acoustic judge (~1.3 GB, kept loaded for a run; fits next to the LLM)."""
    global _scorer
    if _scorer is None:
        import nemo.collections.asr as nemo_asr

        from .podcast.acoustic import AcousticScorer
        from .podcast.nemo_transcriber import NeMoTranscriber

        NeMoTranscriber._quiet_nemo()
        # load on the CPU and halve there: NeMo would load float32 onto the GPU, and the
        # freed float32 copy stays in PyTorch's cache (~2.4 GB) next to the LLM
        model = nemo_asr.models.ASRModel.from_pretrained(settings.nemo_parakeet_model, map_location="cpu")
        _scorer = AcousticScorer(model.eval().half().cuda())
    return _scorer


def release_acoustic_scorer() -> None:
    """Free the acoustic judge: before transcription, and when a run ends (the API server lives on)."""
    global _scorer
    if _scorer is None:
        return
    import gc

    import torch

    _scorer = None
    gc.collect()
    torch.cuda.empty_cache()


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


def _seed_glossaries(episodes: List[PodcastEpisode], results: Dict[str, TranscriptionResult]) -> None:
    """
    Before any episode is fixed, add the speakers named by the publisher's transcripts
    of every episode in this run to their show's glossary as trusted names, so the
    first episodes processed already know the guests of the later ones.
    """
    from .podcast.verifier import glossary_path, load_glossary, save_glossary

    by_show: Dict[str, set] = {}
    for episode in episodes:
        result = results.get(episode.id)
        ref = episode_reference(episode) if result else None
        if ref:
            names = {n.strip() for n in published_speaker_names(ref, result).values() if " " in n.strip()}
            by_show.setdefault(episode.podcast_name or "", set()).update((n, episode.id) for n in names)
    for show, names in by_show.items():
        if not names:
            continue
        path = glossary_path(settings.storage_path, show)
        glossary = load_glossary(path)
        for name, episode_id in names:
            glossary.setdefault(name, {})[episode_id] = "speaker"
        save_glossary(path, glossary)
        logger.info(f"{show}: {len(names)} publisher-confirmed speaker names added to the glossary")


def store_transcript(note_path: Path, check_cancel: Callable[[str], None] = lambda stage: None,
                     index: Optional[dict] = None, link: bool = True):
    """
    Index a transcript note written by save_result and link it into the graph. index: its chunks
    and vectors from the worker; link=False when one rebuild links the whole batch.
    """
    parsed = parse_file(note_path)
    if parsed.status != "success" or not parsed.note:
        raise RuntimeError(f"Could not read back transcript note: {parsed.error_message}")
    check_cancel("index")
    if db.upsert_note(parsed.note, index=index) and link:
        check_cancel("link")
        update_edges_for_note(parsed.note.uuid)
    return parsed.note


def _run_podcast_queue(
    limit: Optional[int] = None,
    enhance: bool = True,
    verify: bool = True,
    sync: bool = True,
    on_episode_start: Optional[Callable[[str, str], None]] = None,
    on_episode_done: Optional[Callable[[EpisodeOutcome], None]] = None,
    progress_for: Optional[Callable[[str], ProgressCallback]] = None,
    on_pass: Optional[Callable[[str, int], None]] = None,
    run=None,
    queue=None,
    store: Optional[Callable[[Path, Callable[[str], None]], object]] = None,
) -> QueueRunSummary:
    """
    Process pending podcast episodes in two passes:

    1. transcribe + diarize every episode (results saved next to the audio as
       <name>.asr.json, so a reprocess skips this pass)
    2. LLM analysis, audio-checked corrections, note, index (LLM + acoustic judge
       loaded once), with the glossary seeded from every episode's publisher transcript

    Args:
        limit: Max episodes to process (None = all pending)
        enhance: Run LLM analysis + cleanup
        verify: Check corrections against the audio (needs enhance)
        sync: First add any audio under storage/podcasts that isn't queued yet
        on_episode_start: Called with (episode_id, title) before each episode, in each pass
        on_episode_done: Called with the outcome of each episode
        progress_for: Returns a progress callback for an episode id; the
            callback returning False cancels that episode
        on_pass: Called with ("transcribe" | "enhance", episode count) as each pass starts
        run: runstate.RunState publishing progress for the process page. A stop request
            (runstate.request_stop) ends the run after the current episode; the rest go
            back to pending
        queue: The jobs to run (default: this machine's TranscriptionQueue; a worker passes
            the jobs it claimed from the primary)
        store: (note path, check_cancel) -> indexes a written note (default: store_transcript;
            a worker's queue sends the note to the primary when the job is marked completed)
    """
    from . import runstate

    from . import embeddings, llm
    from .podcast.asr_cache import load_asr, save_asr
    from .podcast.enhancer import TranscriptEnhancer
    from .podcast.formatter import save_result
    from .podcast.nemo_transcriber import CancellationError, NeMoTranscriber
    from .podcast.queue import TranscriptionQueue

    queue = queue if queue is not None else TranscriptionQueue()
    store = store or store_transcript
    if sync:
        queue.sync_with_storage()

    items = [i for i in queue.items.values() if i.status == "pending"]
    if limit is not None:
        items = items[:limit]
    summary = QueueRunSummary()
    results: Dict[str, TranscriptionResult] = {}
    elapsed: Dict[str, float] = {}

    def title_of(episode: PodcastEpisode) -> str:
        return episode.episode_title or Path(episode.audio_path).stem

    def finish(outcome: EpisodeOutcome, audio_seconds: float = 0.0) -> None:
        summary.outcomes.append(outcome)
        if run:
            run.done(outcome.title, outcome.status, outcome.processing_seconds or 0.0, audio_seconds)
        if on_episode_done:
            on_episode_done(outcome)

    def watched(episode_id: str) -> Optional[ProgressCallback]:
        """The caller's progress callback (cancellation), also reporting stages to the run state."""
        inner = progress_for(episode_id) if progress_for else None
        names = {"chunking": "decode", "transcribing": "asr", "diarizing": "diarize", "aligning": "align"}

        def callback(stage: str, chunk: int, total: int, device: str) -> bool:
            if run and stage in names:
                run.stage(names[stage], chunk, total)
            return inner(stage, chunk, total, device) if inner else True
        return callback

    def fail(episode: PodcastEpisode, error: Exception) -> None:
        outcome = EpisodeOutcome(episode_id=episode.id, title=title_of(episode), status="failed")
        if isinstance(error, CancellationError):
            queue.mark_cancelled(episode.id)
            outcome.status = "cancelled"
        else:
            logger.exception(f"Episode failed: {outcome.title}")
            queue.mark_failed(episode.id, str(error), max_retries=settings.podcast_max_retries)
            outcome.error = str(error)
        finish(outcome)

    # --- Pass 1: transcription (saved results are reused) ---
    for item in items:
        cached = load_asr(Path(item.episode.audio_path))
        if cached:
            results[item.episode.id] = cached
    to_transcribe = [i.episode for i in items if i.episode.id not in results]
    if to_transcribe:
        if on_pass:
            on_pass("transcribe", len(to_transcribe))
        if run:
            run.phase("transcribe", len(to_transcribe))
        # make room on the GPU for ASR: the LLM, the acoustic judge and, in the API
        # server, search models from an earlier search (they reload on use)
        llm.unload()
        release_acoustic_scorer()
        embeddings.release_models(force=True)
        # Parakeet and Sortformer are reloaded per episode (a few seconds): a Parakeet kept
        # loaded across a diarization fails its next decode with an illegal memory access
        transcriber = NeMoTranscriber()
        try:
            for episode in to_transcribe:
                if runstate.stop_requested():
                    break
                if on_episode_start:
                    on_episode_start(episode.id, title_of(episode))
                if run:
                    run.episode(episode.id, title_of(episode))
                queue.mark_started(episode.id)
                started = time.time()
                try:
                    result = transcriber.transcribe(
                        Path(episode.audio_path), episode=episode, progress_callback=watched(episode.id)
                    )
                    save_asr(Path(episode.audio_path), result)
                    results[episode.id] = result
                    elapsed[episode.id] = time.time() - started
                    if run:
                        run.done(title_of(episode), "completed", elapsed[episode.id], count=False)
                except Exception as e:  # one episode must not stop the batch
                    fail(episode, e)
        finally:
            transcriber.cleanup_gpu()

    # --- Pass 2: analysis, verified corrections, note, index ---
    episodes = [i.episode for i in items if i.episode.id in results]
    if runstate.stop_requested():
        queue.release([i.episode.id for i in items])
        return summary
    if not episodes:
        return summary
    if on_pass:
        on_pass("enhance", len(episodes))
    if run:
        run.phase("enhance", len(episodes))
    enhancer = TranscriptEnhancer() if enhance else None
    if enhancer and verify:
        _seed_glossaries(episodes, results)

    for n, episode in enumerate(episodes):
        if runstate.stop_requested():
            released = queue.release([e.id for e in episodes[n:]])
            logger.info(f"Stopped on request; {released} episode(s) back to pending")
            break
        result = results[episode.id]
        title = title_of(episode)
        if on_episode_start:
            on_episode_start(episode.id, title)
        if run:
            run.episode(episode.id, title)
        progress = progress_for(episode.id) if progress_for else None

        def check_cancel(stage: str) -> None:
            if run and stage in ("save", "index", "link"):
                run.stage(stage)
            if progress and progress(stage, 0, 0, "cuda") is False:
                raise CancellationError("Cancelled by user")

        queue.mark_started(episode.id)
        started = time.time()
        try:
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
                    on_stage=run.stage if run else None,
                )
                result.keywords = enhanced["keywords"]
                result.sections = enhanced["sections"]
                result.summary = enhanced["summary"]
                result.key_points = enhanced["key_points"]

            check_cancel("save")
            note_path = save_result(result, episode, enhanced)
            store(note_path, check_cancel)
            if verifier is not None:
                from .podcast.verifier import save_glossary

                save_glossary(glossary_file, verifier.learn(
                    episode.id, list(enhanced["speakers"].values()),
                    verified=enhanced.get("verified_speakers", []),
                    written=f"{title}\n{episode.description or ''}",
                ))
                if verifier.lookup is not None:
                    verifier.lookup.save()

            queue.mark_completed(episode.id, result, note_path)
            finish(EpisodeOutcome(
                episode_id=episode.id,
                title=title,
                status="completed",
                note_path=note_path,
                duration_seconds=result.duration_seconds,
                processing_seconds=elapsed.get(episode.id, 0.0) + time.time() - started,
                speakers=result.speakers_detected,
            ), audio_seconds=result.duration_seconds)
        except Exception as e:
            fail(episode, e)
        finally:
            # hand the episode's activation peaks (judge, embedder) back to the GPU: they'd sit
            # in PyTorch's cache next to the LLM. Safe here: no Parakeet transcribe() in pass 2.
            _empty_cuda_cache()

    return summary


def _empty_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class RunActiveError(RuntimeError):
    """Another process is already running the queue (only one run can use the GPU)."""


def run_podcast_queue(*args, source: str = "cli", link: Optional[Callable[[], object]] = None,
                      **kwargs) -> QueueRunSummary:
    """
    Process pending podcast episodes end to end, keeping the PC awake meanwhile and
    publishing progress to the run state (source: "cli" | "api" | "worker", shown on the process page).
    link: called once after episodes completed (default: rebuild_graph here; a worker asks the
    primary). Arguments: see _run_podcast_queue. Raises RunActiveError if a run is already going.
    """
    from . import embeddings, runstate

    other = runstate.active_run()
    if other and other.get("pid") != os.getpid():
        raise RunActiveError(f"A run is already going ({other.get('source')}, pid {other.get('pid')})")
    run = runstate.RunState(source)
    try:
        with keep_awake(), embeddings.hold():
            try:
                summary = _run_podcast_queue(*args, run=run, **kwargs)
            finally:
                release_acoustic_scorer()
        if summary.completed:
            # each episode was linked as it finished; communities and the k-best union need everything
            from .linker import rebuild_graph

            run.phase("graph")
            try:
                (link or rebuild_graph)()
            except Exception:
                logger.exception("Graph rebuild after the run failed (run `zettlecast graph rebuild`)")
        return summary
    finally:
        run.finish()


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
