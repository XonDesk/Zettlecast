"""
Zettlecast Podcast Transcription Pipeline

High-quality podcast transcription with speaker diarization,
LLM cleanup, keyword extraction, and Zettlecast integration.
"""

from .models import (
    PodcastEpisode,
    TranscriptSegment,
    TranscriptionResult,
    QueueItem,
)
from .base_transcriber import BaseTranscriber, TranscriberConfig, TranscriberCapabilities
from .transcriber_factory import TranscriberFactory
from .transcriber import PodcastTranscriber
from .nemo_transcriber import NeMoTranscriber
from .queue import TranscriptionQueue
from .enhancer import TranscriptEnhancer
from .chunker import AudioChunk, chunk_audio
from .aligner import Word, align_transcription_with_diarization

__all__ = [
    # Models
    "PodcastEpisode",
    "TranscriptSegment",
    "TranscriptionResult",
    "QueueItem",
    # Transcriber Factory (preferred API)
    "TranscriberFactory",
    "BaseTranscriber",
    "TranscriberConfig",
    "TranscriberCapabilities",
    # Legacy transcribers (still available)
    "PodcastTranscriber",
    "NeMoTranscriber",
    # Queue and processing
    "TranscriptionQueue",
    "TranscriptEnhancer",
    "AudioChunk",
    "chunk_audio",
    "Word",
    "align_transcription_with_diarization",
]

