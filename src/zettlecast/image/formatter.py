"""
Markdown Formatter for Vision Results

Formats vision extraction data into structured markdown for note full_text.
"""

from .models import VisionExtraction

# media type -> what to do with it
MEDIA_VERBS = {"book": "to read", "article": "to read", "paper": "to read", "comic": "to read",
               "film": "to watch", "movie": "to watch", "show": "to watch", "series": "to watch",
               "video": "to watch", "course": "to watch", "documentary": "to watch",
               "podcast": "to listen", "album": "to listen", "song": "to listen", "music": "to listen",
               "game": "to play"}
SAVED_TAGS = {"quote": "quote", "idea": "idea", "tip": "tip", "place": "place to visit",
              "product": "to buy", "recipe": "recipe", "reference": "reference"}
TITLE_PREFIX = {"place": "Place", "product": "To buy", "recipe": "Recipe"}


def saved_tag(vision: VisionExtraction) -> str:
    """The tag for what a screenshot was kept as ("to read", "place to visit"...), "" for nothing."""
    if vision.saved_as == "media":
        return MEDIA_VERBS.get(vision.item.get("type", "").lower(), "media")
    return SAVED_TAGS.get(vision.saved_as, "")


def saved_title(vision: VisionExtraction) -> str:
    """'Book: Dune', 'Place: Kamikochi', or the model's title when the item has no name."""
    name = vision.item.get("name", "")
    if vision.saved_as == "media" and name:
        kind = vision.item.get("type", "").strip()
        return f"{kind.capitalize()}: {name}" if kind else name
    if vision.saved_as in TITLE_PREFIX and name:
        return f"{TITLE_PREFIX[vision.saved_as]}: {name}"
    return vision.title


def kept_lines(vision: VisionExtraction) -> list[str]:
    """'## Kept' section: what it was kept as and the thing itself (indexed, so it's searchable)."""
    tag, item = saved_tag(vision), vision.item
    if not tag or not item.get("name"):
        return []
    line = f"**{tag.capitalize()}**: {item['name']}"
    if item.get("type") and vision.saved_as != "media":
        line += f" ({item['type']})"
    if item.get("by"):
        line += f" - {item['by']}"
    lines = ["## Kept", "", line]
    if item.get("where"):
        lines.append(f"- **Where**: {item['where']}")
    if item.get("price"):
        lines.append(f"- **Price**: {item['price']}")
    return lines + [""]


def format_vision_result(
    vision: VisionExtraction,
    width: int,
    height: int,
    file_format: str = "unknown",
) -> str:
    """
    Format vision extraction into searchable markdown.

    A screenshot with a focus (the post / comment it was taken for) keeps everything else on
    the screen in "## Screen Text", which isn't embedded (chunker.UNINDEXED_SECTIONS): the
    other comments and the page around it would otherwise link the note to unrelated topics.
    Without a focus, all the text is indexed. File details are never indexed.
    """
    megapixels = (width * height) / 1_000_000
    aspect_ratio = width / height if height > 0 else 1.0
    text = vision.detected_text.strip()
    has_text = bool(text) and text != "No text detected."

    sections = kept_lines(vision) + ["## Description", "", vision.description, ""]
    if vision.focus:
        sections += ["## Focus", "", vision.focus, ""]
    if vision.context:
        sections += ["## Context", "", vision.context, ""]
    if has_text and not vision.focus:
        sections += ["## Text", "", text, ""]

    sections += ["## Concepts", ""]
    sections += [f"- {c}" for c in vision.concepts] or ["_None detected_"]
    sections.append("")

    if has_text and vision.focus:
        sections += ["## Screen Text", "", text, ""]
    sections += [
        "## Image Details",
        "",
        f"- **Kind**: {vision.kind}",
        f"- **Format**: {file_format.upper()}",
        f"- **Dimensions**: {width} × {height} pixels ({megapixels:.2f} MP, {aspect_ratio:.2f}:1)",
    ]
    return "\n".join(sections)
