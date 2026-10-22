"""
Raw transcription results (words, timings, speakers) kept next to the episode audio.

A queue run transcribes every episode first and fixes them afterwards, so the
result has to outlive the transcription pass; keeping it also means reprocessing
an episode only redoes the LLM/verification pass. The file records which models
made it, and is ignored once those change.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from ..config import settings
from .models import TranscriptionResult

logger = logging.getLogger(__name__)


def asr_path(audio_path: Path) -> Path:
    audio_path = Path(audio_path)
    return audio_path.with_name(f"{audio_path.stem}.asr.json")


def _models() -> dict:
    return {"asr": settings.nemo_parakeet_model, "diarization": settings.nemo_diarization_model}


def save_asr(audio_path: Path, result: TranscriptionResult) -> Path:
    path = asr_path(audio_path)
    data = {"models": _models(), "result": result.model_dump(mode="json")}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def load_asr(audio_path: Path) -> Optional[TranscriptionResult]:
    """The saved result, or None if there is none or it was made by other models."""
    path = asr_path(audio_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("models") != _models():
            logger.info(f"Saved transcription of {Path(audio_path).name} is from other models; redoing it")
            return None
        return TranscriptionResult.model_validate(data["result"])
    except (OSError, ValueError, KeyError) as e:
        logger.warning(f"Unreadable saved transcription {path.name}: {e}")
        return None


def has_asr(audio_path: Path) -> bool:
    """Whether a saved result from the current models exists, from the file's first bytes only
    (the process page asks for every pending episode; the files are megabytes)."""
    try:
        with asr_path(audio_path).open("r", encoding="utf-8") as f:
            head = f.read(1024)
    except OSError:
        return False
    return head.startswith(json.dumps({"models": _models()})[:-1] + ",")


def delete_asr(audio_path: Path) -> bool:
    path = asr_path(audio_path)
    if path.exists():
        path.unlink()
        return True
    return False
