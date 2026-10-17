"""
NeMo Transcription and Diarization (CUDA)

- Parakeet-TDT (local attention) transcribes up to an hour per pass with word timestamps
- Nemotron-3-Diarization (Sortformer, up to 8 speakers) diarizes the whole
  episode in one pass so speaker labels are consistent end to end

Pipeline:
1. Decode the episode to 16 kHz mono WAV
2. Transcribe in NEMO_CHUNK_DURATION_MINUTES pieces
3. Free Parakeet, diarize the full episode
4. Align words to speakers
"""

import importlib.util
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional

# Sortformer calls torch.compile, which needs Triton. Triton isn't shipped for
# Windows, so run in eager mode when it's missing.
if importlib.util.find_spec("triton") is None:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from ..config import settings
from .aligner import Word, align_transcription_with_diarization
from .models import PodcastEpisode, TranscriptSegment, TranscriptionResult

logger = logging.getLogger(__name__)

# (stage, chunk, total_chunks, device) -> False to cancel
ProgressCallback = Callable[[str, int, int, str], bool]

SAMPLE_RATE = 16000


class CancellationError(Exception):
    """Raised when transcription is cancelled by user."""


class NeMoTranscriber:
    """Parakeet-TDT transcription + Nemotron-3-Diarization on CUDA."""

    def __init__(
        self,
        chunk_duration_minutes: Optional[int] = None,
        enable_diarization: bool = True,
    ):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available; NeMo transcription requires an NVIDIA GPU")
        self.device = "cuda"
        self.chunk_duration_minutes = chunk_duration_minutes or settings.nemo_chunk_duration_minutes
        self.enable_diarization = enable_diarization
        self.parakeet_model_id = settings.nemo_parakeet_model
        self.diarization_model_id = settings.nemo_diarization_model
        self._parakeet = None
        self._sortformer = None

    @property
    def name(self) -> str:
        return f"{self.parakeet_model_id} + {self.diarization_model_id}"

    # --- Models ---

    @staticmethod
    def _quiet_nemo() -> None:
        """NeMo logs through its own handler at INFO; keep only errors."""
        from nemo.utils import logging as nemo_logging

        nemo_logging.setLevel(logging.ERROR)

    def _load_parakeet(self):
        if self._parakeet is None:
            import nemo.collections.asr as nemo_asr

            self._quiet_nemo()

            logger.info(f"Loading {self.parakeet_model_id}")
            model = nemo_asr.models.ASRModel.from_pretrained(self.parakeet_model_id)
            model.eval()
            # Local attention keeps memory linear in audio length (full attention is
            # quadratic and fails past ~10 min on 32 GB); an hour fits in ~14 GB.
            model.change_attention_model("rel_pos_local_attn", [256, 256])
            model.change_subsampling_conv_chunking_factor(1)
            self._parakeet = model.to(self.device).half()
        return self._parakeet

    def _load_sortformer(self):
        if self._sortformer is None:
            from nemo.collections.asr.models import SortformerEncLabelModel

            self._quiet_nemo()

            logger.info(f"Loading {self.diarization_model_id}")
            self._sortformer = SortformerEncLabelModel.from_pretrained(self.diarization_model_id)
            self._sortformer.to(self.device).eval()
        return self._sortformer

    def _release(self, attr: str) -> None:
        # Never call torch.cuda.empty_cache() between Parakeet transcribe() calls:
        # it frees buffers the decoder still holds and aborts the process.
        if getattr(self, attr) is not None:
            import gc

            import torch

            setattr(self, attr, None)
            gc.collect()
            torch.cuda.empty_cache()

    def cleanup_gpu(self) -> None:
        """Free all model VRAM."""
        self._release("_parakeet")
        self._release("_sortformer")

    # --- Steps ---

    def _transcribe_wav(self, wav_path: Path, offset: float) -> List[Word]:
        output = self._load_parakeet().transcribe(
            [str(wav_path)], timestamps=True, return_hypotheses=True, verbose=False
        )
        hypothesis = (output[0] if isinstance(output, tuple) else output)[0]
        return [
            Word(text=w["word"], start=w["start"] + offset, end=w["end"] + offset)
            for w in hypothesis.timestamp["word"]
        ]

    def _diarize_wav(self, wav_path: Path) -> str:
        """Diarize a whole file; returns RTTM text."""
        segments = self._load_sortformer().diarize(audio=[str(wav_path)], batch_size=1)[0]
        lines = []
        for seg in segments:
            start, end, speaker = seg.split()
            start, end = float(start), float(end)
            lines.append(f"SPEAKER episode 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>")
        return "\n".join(lines)

    def transcribe(
        self,
        audio_path: Path,
        episode: Optional[PodcastEpisode] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        """Transcribe and diarize one episode."""
        from pydub import AudioSegment

        def report(stage: str, chunk: int = 0, total: int = 0) -> None:
            if progress_callback and progress_callback(stage, chunk, total, self.device) is False:
                raise CancellationError("Cancelled by user")

        start_time = time.time()
        audio_path = Path(audio_path)
        temp_dir = Path(tempfile.mkdtemp(prefix="zettlecast_asr_"))
        logger.info(f"Transcribing {audio_path.name}")

        try:
            report("chunking")
            audio = AudioSegment.from_file(str(audio_path)).set_channels(1).set_frame_rate(SAMPLE_RATE)
            duration = len(audio) / 1000.0
            full_wav = temp_dir / "episode.wav"
            audio.export(str(full_wav), format="wav")

            chunk_ms = self.chunk_duration_minutes * 60 * 1000
            starts = list(range(0, len(audio), chunk_ms))
            words: List[Word] = []
            for i, start_ms in enumerate(starts):
                report("transcribing", i + 1, len(starts))
                if len(starts) == 1:
                    chunk_wav = full_wav
                else:
                    chunk_wav = temp_dir / f"chunk_{i:03d}.wav"
                    audio[start_ms:start_ms + chunk_ms].export(str(chunk_wav), format="wav")
                words.extend(self._transcribe_wav(chunk_wav, offset=start_ms / 1000.0))
            del audio
            self._release("_parakeet")
            logger.info(f"Transcribed {len(words)} words")

            rttm = ""
            if self.enable_diarization:
                report("diarizing", len(starts), len(starts))
                rttm = self._diarize_wav(full_wav)
                self._release("_sortformer")

            report("aligning", len(starts), len(starts))
            if rttm:
                segments = [
                    TranscriptSegment(start=s.start, end=s.end, text=s.text, speaker=s.speaker)
                    for s in align_transcription_with_diarization(words, rttm)
                ]
            else:
                segments = [TranscriptSegment(
                    start=words[0].start if words else 0.0,
                    end=words[-1].end if words else 0.0,
                    text=" ".join(w.text for w in words),
                    speaker=None,
                )]

            processing_time = time.time() - start_time
            logger.info(f"Transcription complete in {processing_time:.1f}s for {duration / 60:.1f} min of audio")

            return TranscriptionResult(
                episode_id=episode.id if episode else "unknown",
                segments=segments,
                full_text=self._format_transcript(segments),
                language="en",
                speakers_detected=len({s.speaker for s in segments if s.speaker}) or 1,
                duration_seconds=duration,
                processing_time_seconds=processing_time,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.cleanup_gpu()

    @staticmethod
    def _format_transcript(segments: List[TranscriptSegment]) -> str:
        lines = []
        for seg in segments:
            speaker = f"{seg.speaker}: " if seg.speaker else ""
            lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {speaker}{seg.text}")
        return "\n".join(lines)
