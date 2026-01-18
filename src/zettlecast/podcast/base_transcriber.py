"""
Base Transcriber Interface

Abstract base class defining the transcriber contract for all platforms.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import PodcastEpisode, TranscriptionResult

logger = logging.getLogger(__name__)


@dataclass
class TranscriberConfig:
    """Configuration for transcriber backends."""
    
    # Device settings
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # Transcription model
    transcription_model: str = "auto"  # auto selects per platform
    
    # Diarization settings
    enable_diarization: bool = True
    diarization_model: str = "auto"  # auto selects per platform
    max_speakers: int = 8
    
    # Audio chunking
    chunk_duration_minutes: int = 10
    
    # HuggingFace token for gated models (pyannote, etc.)
    hf_token: Optional[str] = None
    
    # Additional options
    language: str = "en"
    beam_size: int = 5


@dataclass
class TranscriberCapabilities:
    """Describes what a transcriber backend supports."""
    
    platform: str
    transcriber_name: str
    diarizer_name: Optional[str]
    vad_name: str
    supports_diarization: bool
    supports_gpu: bool
    supports_streaming: bool = False
    requires_container: bool = False


class BaseTranscriber(ABC):
    """
    Abstract base class for all transcriber implementations.
    
    Implementations:
    - MacTranscriber: parakeet-mlx + pyannote.audio
    - NeMoTranscriber: NeMo Parakeet-TDT + TitaNet/MSDD (container)
    - WhisperTranscriber: faster-whisper fallback
    """
    
    def __init__(self, config: Optional[TranscriberConfig] = None):
        self.config = config or TranscriberConfig()
    
    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        episode: Optional[PodcastEpisode] = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            episode: Optional episode metadata
            
        Returns:
            TranscriptionResult with segments and metadata
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> TranscriberCapabilities:
        """
        Return capabilities of this transcriber backend.
        
        Returns:
            TranscriberCapabilities describing what this backend supports
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if this transcriber backend is available on the current system.
        
        Returns:
            True if available, False otherwise
        """
        return True
    
    def warmup(self) -> None:
        """
        Pre-load models to reduce first-transcription latency.
        
        Optional - implementations can override if they support warmup.
        """
        pass
