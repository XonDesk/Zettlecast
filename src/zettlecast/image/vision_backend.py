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


VISION_PROMPT = """This is from the user's own camera roll or screenshots: they kept it to remember
something. Most often a film, show, book, podcast, album, song, video or game to check out; a
quote or saying they like; an idea; a tip or how-to; a place to visit; a product they want; a
recipe. Work out what they kept it for and name that thing exactly (a book shown in a podcast
video is the book; a map pin is the place, not the map app).

Analyse the image and return JSON with:
- "saved_as": what it was kept as: "media" (something to watch, read, listen to or play),
  "quote", "idea", "tip", "place", "product", "recipe", "reference" (facts or a document to look
  up later) or "other" (nothing to keep: a photo of a moment, a conversation)
- "item": the thing kept, all strings, "" where unknown or not applicable:
  "name" (the title, place name, product name or dish; for a podcast episode or video, the
  episode's own title when shown, not just the show; for a quote, its first few words),
  "by" (author, director, artist, show or channel, brand, or who said the quote; not the person
  who merely posted or commented about it),
  "type" (media: book, film, show, podcast, album, song, video, game, article, course;
  place: restaurant, bar, cafe, shop, park, trail, beach, museum, hotel, city, venue...),
  "where" (place: address, area, city, country - as precise as the image or your knowledge allows;
  product: the shop or site), "price" (product: with currency, as shown),
  "why" (one sentence: what is said about the thing - why it's recommended, who or what it's good
  for, any tip that comes with it - written about the thing itself, never about who said it: "Puts
  taper nerves to rest: running less before an ultra is normal", not "A user recommends it")
- "kind": "screenshot" (of an app, website, chat, document or social feed), "photo", or "other"
- "focus": for a screenshot, the one piece of content it was most likely taken for, copied word
  for word: usually a single post, comment, reply, message or paragraph - the one that is
  highlighted, circled, selected or pointed at if anything is, else the one fully in view near
  the middle of the screen. Leave out the other comments and posts, navigation, buttons, ads,
  like counts and timestamps. Empty for photos, and for a screenshot where nothing stands out
  (then "detected_text" carries it).
- "context": for a screenshot, one sentence: the app or site, who wrote the focus (name or
  handle) and what it replies to or sits under. Empty otherwise.
- "title": 4-10 words naming what was kept (the item's name when there is one), for a note title
- "description": 2-3 specific sentences on the substance: what the thing, idea or tip is and what
  is said about it. Never describe the screenshot, the app or who posted it ("A Reddit comment...",
  "a user suggests..."): "context" has that. For a photo: the main subject, setting and details.
- "detected_text": the readable text, preserving line breaks; "No text detected." if there is
  none. For a photo or a document: all of it (signs, labels, captions, pages). For a screenshot:
  the content, not the app - one block per post, comment or message, blocks separated by a blank
  line, the author's name or handle alone on the block's first line (add " OP" if marked), each
  paragraph on one line rather than broken where the screen wraps it. Leave out the status bar
  (clock, battery, network), buttons and prompts (Reply, Share, Follow, Add a comment, Send
  message), vote, like, view and comment counts, "17h" / "2 days ago" times, and ads.
- "concepts": 5-7 lowercase key topics of the focus content (or the image). No names of people or
  organisations.
- "people": full names of people who matter to the focus content (or shown in the image)
- "organisations": organisations, companies, products, races or events the focus content names"""

SAVED_AS = ["media", "quote", "idea", "tip", "place", "product", "recipe", "reference", "other"]
ITEM_FIELDS = ["name", "by", "type", "where", "price", "why"]

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "saved_as": {"type": "string", "enum": SAVED_AS},
        "item": {"type": "object", "properties": {f: {"type": "string"} for f in ITEM_FIELDS},
                 "required": ITEM_FIELDS},
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
    "required": ["saved_as", "item", "kind", "focus", "context", "title", "description", "detected_text", "concepts",
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
        item = result.get("item") if isinstance(result.get("item"), dict) else {}
        item = {f: str(item[f]).strip() for f in ITEM_FIELDS if str(item.get(f) or "").strip()}
        saved_as = result.get("saved_as") if result.get("saved_as") in SAVED_AS else "other"
        focus = (result.get("focus") or "").strip()
        if focus:
            # names only from the content the screenshot was taken for, not the ads or the other
            # comments around it (the model lists those too)
            about = f"{focus} {result.get('context') or ''} {item.get('name', '')} {item.get('by', '')}".lower()
            for field in ("people", "organisations"):
                result[field] = [n for n in result.get(field) or []
                                 if isinstance(n, str) and n.strip().lower() in about]
        return VisionExtraction(
            saved_as=saved_as,
            item=item,
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
