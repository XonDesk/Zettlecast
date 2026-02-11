"""
Image Parser

Main entry point for image processing - orchestrates vision analysis and note creation.
"""

import hashlib
import logging
import time
from pathlib import Path
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


def parse_image(file_path: Path) -> ProcessingResult:
    """
    Parse image file using Qwen2.5-VL vision model.

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
            backend = VisionBackend(
                model=settings.vision_model,
                base_url=settings.ollama_base_url,
            )
            vision_data = backend.analyze_image(file_path)
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
        title = file_path.stem.replace("_", " ").replace("-", " ").title()

        note = NoteModel(
            uuid=note_uuid,
            title=title,
            source_type="image",
            source_path=str(file_path.absolute()),
            full_text=full_text,
            content_hash=compute_content_hash(full_text),
            metadata=NoteMetadata(
                tags=vision_data.concepts,  # Use concepts as tags
                custom={
                    "image_width": width,
                    "image_height": height,
                    "megapixels": round(megapixels, 2),
                    "image_format": file_format,
                    "detected_text": vision_data.detected_text,
                    "description": vision_data.description,
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
