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

    Creates a structured document with:
    - Visual description
    - OCR text
    - Detected concepts
    - Image metadata

    This format makes the full_text field semantically rich for embeddings.

    Args:
        vision: Vision extraction results
        width: Image width in pixels
        height: Image height in pixels
        file_format: Image format (PNG, JPEG, etc.)

    Returns:
        Formatted markdown string
    """
    megapixels = (width * height) / 1_000_000
    aspect_ratio = width / height if height > 0 else 1.0

    sections = [
        "# Visual Description",
        "",
        vision.description,
        "",
        "# Detected Text (OCR)",
        "",
        vision.detected_text if vision.detected_text.strip() else "_No text detected_",
        "",
        "# Detected Objects & Concepts",
        "",
    ]

    # Add concepts as bullet list
    if vision.concepts:
        for concept in vision.concepts:
            sections.append(f"- {concept}")
    else:
        sections.append("_No specific objects detected_")

    # Add metadata section
    sections.extend([
        "",
        "# Image Metadata",
        "",
        f"- **Format**: {file_format.upper()}",
        f"- **Dimensions**: {width} × {height} pixels",
        f"- **Megapixels**: {megapixels:.2f} MP",
        f"- **Aspect Ratio**: {aspect_ratio:.2f}:1",
    ])

    return "\n".join(sections)
