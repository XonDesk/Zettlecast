"""
Image Parser

Main entry point for image processing - orchestrates vision analysis and note creation.
"""

import hashlib
import logging
import shutil
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from ..config import settings
from ..models import NoteMetadata, NoteModel, ProcessingResult
from .formatter import format_vision_result
from .vision_backend import VisionBackend, VisionModelNotFoundError

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def keep_copy(file_path: Path) -> Path:
    """A copy of the image in <storage>/images (named by content, so a re-run reuses it)."""
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()[:12]
    target = settings.storage_path / "images" / f"{digest}{file_path.suffix.lower()}"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target)
    return target


def parse_image(file_path: Path, keep_image: bool = True, focus_region: Optional[dict] = None) -> ProcessingResult:
    """
    Parse image file using the configured vision model.

    keep_image=False (screenshot triage "keep"): only the information is kept; the note records
    where the screenshot was but doesn't show or serve it. Screenshot triage "keep + image" keeps
    a copy in storage, so the note still has it when the original is deleted.

    Complete pipeline:
    1. Validate file and check size
    2. Extract image dimensions
    3. Run vision analysis (description + OCR + concepts)
    4. Format as markdown
    5. Create NoteModel with metadata

    Args:
        file_path: Path to image file

    Returns:
        ProcessingResult with success/failure status and NoteModel
    """
    start_time = time.time()

    try:
        # 1. Validate file exists
        if not file_path.exists():
            return ProcessingResult(
                status="failed",
                error_type="parse",
                error_message=f"File not found: {file_path}",
            )

        # 2. Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > settings.max_image_size_mb:
            return ProcessingResult(
                status="failed",
                error_type="parse",
                error_message=f"Image too large: {size_mb:.1f}MB (max: {settings.max_image_size_mb}MB)",
            )

        # 3. Get image dimensions and format
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                file_format = img.format or file_path.suffix[1:].upper()
                megapixels = (width * height) / 1_000_000
        except UnidentifiedImageError:
            return ProcessingResult(
                status="failed",
                error_type="parse",
                error_message="Corrupted or unsupported image format",
            )

        logger.info(f"Processing image: {file_path.name} ({width}x{height}, {megapixels:.2f}MP)")

        # 4. Run vision analysis
        try:
            vision_data = VisionBackend().analyze_image(file_path, focus_region=focus_region)
        except VisionModelNotFoundError as e:
            return ProcessingResult(
                status="failed",
                error_type="parse",
                error_message=str(e),
            )

        # 5. Format as markdown
        full_text = format_vision_result(vision_data, width, height, file_format)

        # 6. Create note with UUID
        note_uuid = str(uuid4())
        title = vision_data.title or file_path.stem.replace("_", " ").replace("-", " ").title()
        source = file_path.absolute()
        if keep_image and _is_triage_screenshot(file_path):
            source = keep_copy(file_path)

        note = NoteModel(
            uuid=note_uuid,
            title=title,
            source_type="image",
            source_path=str(source),
            full_text=full_text,
            content_hash=compute_content_hash(full_text),
            metadata=NoteMetadata(
                tags=vision_data.concepts,  # Use concepts as tags
                custom={
                    "image_width": width,
                    "image_height": height,
                    "megapixels": round(megapixels, 2),
                    "image_format": file_format,
                    "image_kept": keep_image,
                    "original_path": str(file_path.absolute()),
                    "kind": vision_data.kind,
                    **({"focus_region": focus_region} if focus_region else {}),
                    **({"focus": vision_data.focus} if vision_data.focus else {}),
                    **({"context": vision_data.context} if vision_data.context else {}),
                    "detected_text": vision_data.detected_text,
                    "description": vision_data.description,
                    **({"people": vision_data.people} if vision_data.people else {}),
                    **({"organisations": vision_data.organisations} if vision_data.organisations else {}),
                },
            ),
        )

        processing_time = time.time() - start_time
        logger.info(f"Image processed in {processing_time:.1f}s: {note_uuid[:8]}")

        return ProcessingResult(
            status="success",
            uuid=note_uuid,
            title=title,
            note=note,
        )

    except Exception as e:
        logger.exception(f"Image parsing failed: {e}")
        return ProcessingResult(
            status="failed",
            error_type="parse",
            error_message=str(e),
        )


def _is_triage_screenshot(file_path: Path) -> bool:
    """Screenshots from a triage folder live outside storage (e.g. OneDrive) and may be deleted."""
    from .triage import in_folders

    return in_folders(file_path)
