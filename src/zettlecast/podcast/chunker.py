"""
Audio Chunking Module

Splits audio files into fixed-duration chunks (10 minutes) for stable memory usage
during transcription and diarization.
"""

import logging
import tempfile
from pathlib import Path
from typing import Generator, Tuple

logger = logging.getLogger(__name__)


class AudioChunk:
    """Represents a chunk of audio with timing information."""

    def __init__(self, path: Path, start_time: float, end_time: float, chunk_index: int):
        self.path = path
        self.start_time = start_time  # Offset in original file (seconds)
        self.end_time = end_time
        self.chunk_index = chunk_index
        self.duration = end_time - start_time

    def __repr__(self):
        return (
            f"AudioChunk(index={self.chunk_index}, "
            f"start={self.start_time:.1f}s, end={self.end_time:.1f}s)"
        )


def chunk_audio(
    audio_path: Path,
    chunk_duration_minutes: int = 10,
    temp_dir: Path = None,
) -> Generator[AudioChunk, None, None]:
    """
    Split audio file into fixed-duration chunks.

    Args:
        audio_path: Path to input audio file
        chunk_duration_minutes: Duration of each chunk in minutes
        temp_dir: Directory for temporary chunk files (default: system temp)

    Yields:
        AudioChunk objects with temporary file paths and timing info

    Note:
        Caller is responsible for cleaning up temporary chunk files.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        logger.error("pydub is required for audio chunking. Install: pip install pydub")
        raise

    logger.info(f"Loading audio: {audio_path.name}")
    audio = AudioSegment.from_file(str(audio_path))
    total_duration_ms = len(audio)
    total_duration_sec = total_duration_ms / 1000.0

    chunk_duration_ms = chunk_duration_minutes * 60 * 1000
    num_chunks = (total_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms

    logger.info(
        f"Splitting {total_duration_sec:.1f}s audio into {num_chunks} chunks "
        f"of {chunk_duration_minutes} minutes"
    )

    # Create temp directory if not provided
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="zettlecast_chunks_"))
    else:
        temp_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_chunks):
        start_ms = i * chunk_duration_ms
        end_ms = min((i + 1) * chunk_duration_ms, total_duration_ms)

        # Extract chunk
        chunk_audio = audio[start_ms:end_ms]

        # Save to temp file
        chunk_path = temp_dir / f"chunk_{i:03d}.wav"
        chunk_audio.export(str(chunk_path), format="wav")

        # Create AudioChunk object
        chunk = AudioChunk(
            path=chunk_path,
            start_time=start_ms / 1000.0,
            end_time=end_ms / 1000.0,
            chunk_index=i,
        )

        logger.debug(f"Created {chunk}")
        yield chunk


def get_audio_duration(audio_path: Path) -> float:
    """
    Get audio duration in seconds.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds
    """
    try:
        import torchaudio

        info = torchaudio.info(str(audio_path))
        return info.num_frames / info.sample_rate
    except Exception:
        # Fallback to pydub
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(str(audio_path))
            return len(audio) / 1000.0
        except Exception as e:
            logger.error(f"Could not get duration for {audio_path}: {e}")
            return 0.0
