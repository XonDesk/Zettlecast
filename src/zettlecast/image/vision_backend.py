"""
Vision Backend for Image Analysis

One structured Ollama call per image returns a description, the visible
text (OCR) and concept tags.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

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


REGION_PROMPT = """The user circled part of this screenshot. Image 1 is the whole screenshot with
their mark drawn on it in magenta; image 2 is a close-up of the circled part. The focus is the
circled content: copy it word for word from image 2 (a rough circle can clip a few edge words:
complete them from image 1). Use the rest of image 1 only for the context.

"""

REGION_PAD = 0.04  # of the image size, around the circled box: rough circles cut corners
MIN_CROP_PX = 600  # small crops are enlarged so the model can read them


def region_images(image_path: Path, region: dict, workdir: Path) -> list[Path]:
    """The screenshot with the user's mark drawn on it, and a close-up of the circled part."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        marked = img.copy()
        points = [(x * w, y * h) for x, y in region.get("points") or []]
        if len(points) > 1:
            ImageDraw.Draw(marked).line(points + points[:1], fill=(255, 0, 200), width=max(3, w // 200))
        x, y, bw, bh = region["box"]
        box = (int(max(0.0, x - REGION_PAD) * w), int(max(0.0, y - REGION_PAD) * h),
               int(min(1.0, x + bw + REGION_PAD) * w), int(min(1.0, y + bh + REGION_PAD) * h))
        crop = img.crop(box)
        scale = MIN_CROP_PX / max(1, min(crop.size))
        if scale > 1:
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
    paths = [workdir / "marked.png", workdir / "circled.png"]
    marked.save(paths[0])
    crop.save(paths[1])
    return paths


class VisionBackend:
    """Image understanding via the configured Ollama vision model."""

    def __init__(self, model: str = None):
        self.model = model or settings.vision_model

    def analyze_image(self, image_path: Path, focus_region: Optional[dict] = None) -> VisionExtraction:
        """focus_region: the part the user circled (fractions of the image), read as the focus."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Analyzing image with {self.model}: {image_path.name}"
                    + (" (circled region)" if focus_region else ""))
        with tempfile.TemporaryDirectory() as tmp:
            if focus_region:
                prompt, images = REGION_PROMPT + VISION_PROMPT, region_images(image_path, focus_region, Path(tmp))
            else:
                prompt, images = VISION_PROMPT, [image_path]
            try:
                result = chat(prompt, schema=VISION_SCHEMA, images=images, model=self.model)
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
