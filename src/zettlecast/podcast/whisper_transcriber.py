"""
Whisper Transcriber - CPU Fallback with Diarization

Uses faster-whisper for transcription on any platform.
Optional pyannote.audio diarization for speaker detection.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

from .base_transcriber import BaseTranscriber, TranscriberCapabilities, TranscriberConfig
from .diarization_mixin import DiarizationMixin
from .models import PodcastEpisode, TranscriptSegment, TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperTranscriber(BaseTranscriber, DiarizationMixin):
    """
    Transcriber using faster-whisper with optional pyannote diarization.
    
    Works on all platforms. Supports diarization via pyannote.audio
    when enabled and HuggingFace token is provided.
    """
    
    def __init__(self, config: Optional[TranscriberConfig] = None):
        super().__init__(config)
        
        # Model loaded lazily
        self._whisper_model = None
        
        # Model size from config or default
        self.model_size = getattr(
            self.config, 'transcription_model', None
        ) or "large-v3-turbo"
        
        if self.model_size == "auto":
            self.model_size = "large-v3-turbo"
    
    def is_available(self) -> bool:
        """Check if faster-whisper is available."""
        try:
            import faster_whisper
            return True
        except ImportError:
            return False
    
    def _load_whisper(self):
        """Load faster-whisper model."""
        if self._whisper_model is None:
            from faster_whisper import WhisperModel
            
            # Detect best device
            device = self.config.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            
            # Optimize compute type for device
            if device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"
            
            logger.info(f"Loading Whisper model: {self.model_size} on {device}")
            self._whisper_model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
            )
            logger.info("Whisper model loaded")
        
        return self._whisper_model
    
    def warmup(self) -> None:
        """Pre-load model."""
        self._load_whisper()
        if self.config.enable_diarization:
            self._load_diarization_pipeline()
    
    def get_capabilities(self) -> TranscriberCapabilities:
        """Return Whisper transcriber capabilities."""
        diarization_available = (
            self.config.enable_diarization and 
            self._is_diarization_available() and 
            bool(self.config.hf_token)
        )
        return TranscriberCapabilities(
            platform="any",
            transcriber_name="faster-whisper",
            diarizer_name="pyannote.audio" if diarization_available else None,
            vad_name="silero",  # faster-whisper uses Silero VAD internally
            supports_diarization=True,  # Now supports diarization!
            supports_gpu=True,  # Can use CUDA if available
            supports_streaming=False,
            requires_container=False,
        )
    
    def transcribe(
        self,
        audio_path: Path,
        episode: Optional[PodcastEpisode] = None,
        progress_callback=None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using faster-whisper with optional diarization.
        
        Args:
            audio_path: Path to audio file
            episode: Optional episode metadata
            
        Returns:
            TranscriptionResult with segments and speaker labels if diarization enabled
        """
        start_time = time.time()
        audio_path = Path(audio_path)
        
        logger.info(f"Starting Whisper transcription: {audio_path.name}")
        
        # Load and transcribe with word timestamps for better diarization
        whisper = self._load_whisper()
        segments_raw, info = whisper.transcribe(
            str(audio_path),
            beam_size=self.config.beam_size,
            language=self.config.language,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400,
            },
            word_timestamps=self.config.enable_diarization,  # Enable for diarization
        )
        
        # Get duration from info
        duration = info.duration
        logger.info(f"Audio duration: {duration:.1f}s ({duration/60:.1f} min)")
        
        # Convert generator to list
        segments_list = list(segments_raw)
        logger.info(f"Transcribed {len(segments_list)} segments")
        
        # Build transcript segments
        transcript_segments = []
        speakers_detected = 1
        
        if self.config.enable_diarization and self._is_diarization_available():
            # Extract words for diarization alignment
            from .aligner import Word
            words = []
            for seg in segments_list:
                if hasattr(seg, 'words') and seg.words:
                    for word in seg.words:
                        words.append(Word(
                            text=word.word.strip(),
                            start=word.start,
                            end=word.end,
                        ))
            
            if words:
                # Apply diarization with word-level alignment
                transcript_segments, speakers_detected = self._apply_diarization_to_words(
                    audio_path, words
                )
            else:
                # Fallback to segment-level diarization
                for seg in segments_list:
                    transcript_segments.append(
                        TranscriptSegment(
                            start=seg.start,
                            end=seg.end,
                            text=seg.text.strip(),
                            speaker=None,
                        )
                    )
                transcript_segments, speakers_detected = self._apply_diarization_to_segments(
                    audio_path, transcript_segments
                )
        else:
            # No diarization - simple segment extraction
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
            language=info.language if hasattr(info, "language") else self.config.language,
            speakers_detected=speakers_detected,
            duration_seconds=duration,
            processing_time_seconds=processing_time,
        )
    
    def _format_transcript(self, segments: List[TranscriptSegment]) -> str:
        """Format segments into readable transcript."""
        lines = []
        for seg in segments:
            timestamp = f"[{seg.start:.1f}s]"
            speaker = f"{seg.speaker}: " if seg.speaker else ""
            lines.append(f"{timestamp} {speaker}{seg.text}")
        
        return "\n".join(lines)

