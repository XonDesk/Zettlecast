"""
Markdown Formatter for Vision Results

Formats vision extraction data into structured markdown for note full_text.
"""

from .models import VisionExtraction


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

    sections = ["## Description", "", vision.description, ""]
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
