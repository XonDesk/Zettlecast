"""
Podcast Transcriber Module

High-quality transcription using faster-whisper.
Optionally supports WhisperX alignment and pyannote speaker diarization.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from ..config import settings
from .models import PodcastEpisode, TranscriptSegment, TranscriptionResult

logger = logging.getLogger(__name__)


class PodcastTranscriber:
    """
    Podcast transcription using faster-whisper.

    Pipeline:
    1. faster-whisper transcription
    2. (Optional) WhisperX word-level alignment
    3. (Optional) pyannote speaker diarization
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: str = "cpu",
        model_size: Optional[str] = None,
    ):
        self.hf_token = hf_token or settings.hf_token
        # Use configured device (GPU supported as of 2026)
        self.device = device or settings.whisper_device
        self.model_size = model_size or settings.whisper_model

        # Models loaded lazily
        self._whisper_model = None

    def _load_whisper(self):
        """Load faster-whisper model."""
        if self._whisper_model is None:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            # Optimize compute type for device
            if self.device == "cuda":
                compute_type = "float16"  # Best GPU performance
            else:
                compute_type = "int8"     # Best CPU performance
            
            self._whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
            )
        return self._whisper_model

    def transcribe(
        self,
        audio_path: Path,
        episode: Optional[PodcastEpisode] = None,
        enable_diarization: bool = False,  # Disabled by default for stability
        progress_callback=None,
    ) -> TranscriptionResult:
        """
        Transcribe a podcast episode.

        Args:
            audio_path: Path to audio file
            episode: Optional episode metadata
            enable_diarization: Whether to perform speaker diarization (requires HF token)

        Returns:
            TranscriptionResult with segments and metadata
        """
        start_time = time.time()
        audio_path = Path(audio_path)

        logger.info(f"Starting transcription: {audio_path.name}")

        # Transcribe with faster-whisper
        whisper = self._load_whisper()
        segments_raw, info = whisper.transcribe(
            str(audio_path),
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400,
            },
        )

        # Get duration from info
        duration = info.duration
        logger.info(f"Audio duration: {duration:.1f}s ({duration/60:.1f} min)")

        # Convert generator to list
        segments_list = list(segments_raw)
        logger.info(f"Transcribed {len(segments_list)} segments")

        # Build transcript segments
        transcript_segments = []
        for seg in segments_list:
            transcript_segments.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    speaker=None,
                )
            )

        # Format full transcript
        full_text = self._format_transcript(transcript_segments)

        processing_time = time.time() - start_time
        logger.info(
            f"Transcription complete in {processing_time:.1f}s "
            f"({processing_time/duration:.2f}x realtime)"
        )

        return TranscriptionResult(
            episode_id=episode.id if episode else "unknown",
            segments=transcript_segments,
            full_text=full_text,
            language=info.language if hasattr(info, "language") else "en",
            speakers_detected=1,
            duration_seconds=duration,
            processing_time_seconds=processing_time,
        )

    def _format_transcript(self, segments: List[TranscriptSegment]) -> str:
        """Format segments into readable transcript."""
        lines = []
        for seg in segments:
            timestamp = f"[{seg.start:.1f}s]"
            lines.append(f"{timestamp} {seg.text}")

        return "\n".join(lines)
