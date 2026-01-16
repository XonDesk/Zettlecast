"""
Voice Activity Detection (VAD) Module

Uses Silero VAD to preprocess audio and strip non-speech segments,
reducing hallucinations in the transcription pipeline.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)

# Silero VAD model (loaded lazily)
_vad_model = None
_vad_utils = None


def _load_vad_model():
    """Lazily load Silero VAD model."""
    global _vad_model, _vad_utils

    if _vad_model is None:
        torch.set_num_threads(1)  # Silero is optimized for single thread
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        logger.info("Loaded Silero VAD model")

    return _vad_model, _vad_utils


def get_speech_timestamps(audio_path: Path) -> List[Tuple[float, float]]:
    """
    Detect speech segments in audio file.

    Args:
        audio_path: Path to audio file (any format supported by torchaudio)

    Returns:
        List of (start_seconds, end_seconds) tuples for speech segments
    """
    model, utils = _load_vad_model()
    (get_speech_ts, _, read_audio, _, _) = utils

    # Read audio at 16kHz (required by Silero)
    wav = read_audio(str(audio_path), sampling_rate=16000)

    # Get speech timestamps
    speech_timestamps = get_speech_ts(
        wav,
        model,
        sampling_rate=16000,
        min_silence_duration_ms=500,  # Merge segments with <500ms silence
        min_speech_duration_ms=250,   # Ignore very short speech
    )

    # Convert to seconds
    segments = [
        (ts["start"] / 16000, ts["end"] / 16000)
        for ts in speech_timestamps
    ]

    logger.info(f"Found {len(segments)} speech segments in {audio_path.name}")
    return segments


def has_speech(audio_path: Path, min_speech_ratio: float = 0.1) -> bool:
    """
    Check if audio file contains enough speech to be worth transcribing.

    Args:
        audio_path: Path to audio file
        min_speech_ratio: Minimum ratio of speech to total duration

    Returns:
        True if audio contains sufficient speech
    """
    try:
        segments = get_speech_timestamps(audio_path)
        if not segments:
            return False

        total_speech = sum(end - start for start, end in segments)
        total_duration = segments[-1][1] if segments else 0

        if total_duration == 0:
            return False

        ratio = total_speech / total_duration
        return ratio >= min_speech_ratio

    except Exception as e:
        logger.warning(f"VAD check failed for {audio_path}: {e}")
        return True  # Assume has speech if check fails


def get_audio_duration(audio_path: Path) -> float:
    """Get audio file duration in seconds."""
    try:
        import torchaudio
        info = torchaudio.info(str(audio_path))
        return info.num_frames / info.sample_rate
    except Exception:
        # Fallback using pydub
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(audio_path))
            return len(audio) / 1000.0
        except Exception as e:
            logger.error(f"Could not get duration for {audio_path}: {e}")
            return 0.0
