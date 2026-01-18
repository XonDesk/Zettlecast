"""
MLX Whisper Transcriber - Apple Silicon Optimized with Diarization

Uses mlx-whisper for fast transcription on Apple Silicon Macs.
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


class MLXWhisperTranscriber(BaseTranscriber, DiarizationMixin):
    """
    Apple Silicon optimized transcriber using mlx-whisper with diarization.
    
    Very fast transcription using Apple's MLX framework.
    Supports diarization via pyannote.audio when HuggingFace token is provided.
    """
    
    def __init__(self, config: Optional[TranscriberConfig] = None):
        super().__init__(config)
        
        # Default model - can be overridden via config or settings, handle 'auto'
        from ..config import settings
        model_config = getattr(self.config, 'transcription_model', None)
        if model_config and model_config != "auto":
            self.model_id = model_config
        else:
            self.model_id = getattr(settings, 'mlx_whisper_model', 'mlx-community/whisper-large-v3-turbo')
    
    def is_available(self) -> bool:
        """Check if mlx-whisper is available."""
        try:
            import mlx_whisper
            return True
        except ImportError:
            return False
    
    def warmup(self) -> None:
        """Pre-download the model if needed."""
        if not self.is_available():
            raise RuntimeError("mlx-whisper is not installed")
        
        import mlx_whisper
        logger.info(f"MLX Whisper ready with model: {self.model_id}")
        
        if self.config.enable_diarization:
            self._load_diarization_pipeline()
    
    def get_capabilities(self) -> TranscriberCapabilities:
        """Return MLX Whisper transcriber capabilities."""
        diarization_available = (
            self.config.enable_diarization and 
            self._is_diarization_available() and 
            bool(self.config.hf_token)
        )
        return TranscriberCapabilities(
            platform="darwin",
            transcriber_name="mlx-whisper",
            diarizer_name="pyannote.audio" if diarization_available else None,
            vad_name="silero",  # mlx-whisper uses Silero VAD internally
            supports_diarization=True,  # Now supports diarization!
            supports_gpu=True,  # Apple Silicon Neural Engine
            supports_streaming=False,
            requires_container=False,
        )
    
    def transcribe(
        self,
        audio_path: Path,
        episode: Optional[PodcastEpisode] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using mlx-whisper with optional diarization.
        
        Args:
            audio_path: Path to audio file
            episode: Optional episode metadata
            
        Returns:
            TranscriptionResult with segments and speaker labels if diarization enabled
        """
        start_time = time.time()
        audio_path = Path(audio_path)
        
        logger.info(f"Starting MLX Whisper transcription: {audio_path.name}")
        logger.info(f"Using model: {self.model_id}")
        
        import mlx_whisper
        
        # Transcribe with mlx-whisper (word timestamps for diarization)
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self.model_id,
            language=self.config.language if self.config.language != "auto" else None,
            word_timestamps=True,
        )
        
        # Calculate duration from last segment
        raw_segments = result.get("segments", [])
        duration = raw_segments[-1]["end"] if raw_segments else 0.0
        
        # Build transcript segments with optional diarization
        transcript_segments = []
        speakers_detected = 1
        
        if self.config.enable_diarization and self._is_diarization_available():
            # Extract words for diarization alignment
            from .aligner import Word
            words = []
            for segment in raw_segments:
                if "words" in segment:
                    for word_data in segment["words"]:
                        words.append(Word(
                            text=word_data.get("word", "").strip(),
                            start=word_data.get("start", 0.0),
                            end=word_data.get("end", 0.0),
                        ))
            
            if words:
                # Apply diarization with word-level alignment
                transcript_segments, speakers_detected = self._apply_diarization_to_words(
                    audio_path, words
                )
            else:
                # Fallback to segment-level diarization
                for segment in raw_segments:
                    transcript_segments.append(
                        TranscriptSegment(
                            start=segment["start"],
                            end=segment["end"],
                            text=segment["text"].strip(),
                            speaker=None,
                        )
                    )
                transcript_segments, speakers_detected = self._apply_diarization_to_segments(
                    audio_path, transcript_segments
                )
        else:
            # No diarization - simple segment extraction
            for segment in raw_segments:
                transcript_segments.append(
                    TranscriptSegment(
                        start=segment["start"],
                        end=segment["end"],
                        text=segment["text"].strip(),
                        speaker=None,
                    )
                )
        
        # Full text
        full_text = self._format_transcript(transcript_segments)
        
        processing_time = time.time() - start_time
        
        if duration > 0:
            logger.info(
                f"Transcription complete in {processing_time:.1f}s "
                f"({processing_time/duration:.2f}x realtime)"
            )
        else:
            logger.info(f"Transcription complete in {processing_time:.1f}s")
        
        return TranscriptionResult(
            episode_id=episode.id if episode else "unknown",
            segments=transcript_segments,
            full_text=full_text,
            language=result.get("language", self.config.language),
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

