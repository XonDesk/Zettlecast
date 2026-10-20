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
- "kind": "screenshot" (of an app, website, chat, document or social feed), "photo", or "other"
- "focus": for a screenshot, the one piece of content it was most likely taken for, copied word
  for word: usually a single post, comment, reply, message or paragraph - the one that is
  highlighted, circled, selected or pointed at if anything is, else the one fully in view near
  the middle of the screen. Leave out the other comments and posts, navigation, buttons, ads,
  like counts and timestamps. Empty for photos, and for a screenshot where nothing stands out
  (then "detected_text" carries it).
- "context": for a screenshot, one sentence: the app or site, who wrote the focus (name or
  handle) and what it replies to or sits under. Empty otherwise.
- "title": 4-10 words saying what the focus (or the image) is about, for a note title
- "description": 2-3 specific, factual sentences about the focus content, or for a photo the main
  subject, setting and notable details
- "detected_text": ALL visible text (signs, documents, UI, captions), preserving line breaks;
  "No text detected." if there is none
- "concepts": 5-7 lowercase key topics of the focus content (or the image). No names of people or
  organisations.
- "people": full names of people who matter to the focus content (or shown in the image)
- "organisations": organisations, companies, products, races or events the focus content names"""

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["screenshot", "photo", "other"]},
        "focus": {"type": "string"},
        "context": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "detected_text": {"type": "string"},
        "concepts": {"type": "array", "items": {"type": "string"}},
        "people": {"type": "array", "items": {"type": "string"}},
        "organisations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["kind", "focus", "context", "title", "description", "detected_text", "concepts",
                 "people", "organisations"],
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
        focus = (result.get("focus") or "").strip()
        if focus:
            # names only from the content the screenshot was taken for, not the ads or the other
            # comments around it (the model lists those too)
            about = f"{focus} {result.get('context') or ''}".lower()
            for field in ("people", "organisations"):
                result[field] = [n for n in result.get(field) or []
                                 if isinstance(n, str) and n.strip().lower() in about]
        return VisionExtraction(
            kind=result.get("kind") or "other",
            focus=(result.get("focus") or "").strip(),
            context=(result.get("context") or "").strip(),
            title=(result.get("title") or "").strip(),
            description=result.get("description", "").strip(),
            detected_text=result.get("detected_text", "").strip(),
            concepts=concepts[:7],
            people=[p.strip() for p in result.get("people") or [] if isinstance(p, str) and " " in p.strip()],
            organisations=[o.strip() for o in result.get("organisations") or [] if isinstance(o, str) and o.strip()],
        )
