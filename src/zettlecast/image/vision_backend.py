"""
Vision Backend for Image Analysis

One structured Ollama call per image returns a description, the visible
text (OCR) and concept tags.
"""

import logging
from pathlib import Path

from ..config import settings
from ..llm import LLMError, chat
from .models import VisionExtraction

logger = logging.getLogger(__name__)


class VisionModelNotFoundError(Exception):
    """Raised when the vision model can't be used (missing or Ollama down)."""


VISION_PROMPT = """Analyse this image and return JSON with:
- "description": 2-3 specific, factual sentences about the main subject, setting and notable details
- "detected_text": ALL visible text (signs, documents, UI, captions), preserving line breaks;
  "No text detected." if there is none
- "concepts": 5-7 lowercase key objects, concepts or themes"""

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "detected_text": {"type": "string"},
        "concepts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["description", "detected_text", "concepts"],
}


class VisionBackend:
    """Image understanding via the configured Ollama vision model."""

    def __init__(self, model: str = None):
        self.model = model or settings.vision_model

    def analyze_image(self, image_path: Path) -> VisionExtraction:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Analyzing image with {self.model}: {image_path.name}")
        try:
            result = chat(VISION_PROMPT, schema=VISION_SCHEMA, images=[image_path], model=self.model)
        except LLMError as e:
            raise VisionModelNotFoundError(str(e)) from e

        concepts = [str(c).lower().strip() for c in result.get("concepts", []) if str(c).strip()]
        return VisionExtraction(
            description=result.get("description", "").strip(),
            detected_text=result.get("detected_text", "").strip(),
            concepts=concepts[:7],
        )
