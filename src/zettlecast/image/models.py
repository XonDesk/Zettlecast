"""
Image Processing Data Models

Data structures for image queue management and vision processing results.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ImageItem(BaseModel):
    """
    Represents an image in the processing queue.

    Similar to PodcastEpisode in the podcast module.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    image_path: str
    image_hash: str  # SHA256 for deduplication
    collection_name: Optional[str] = None  # Like podcast_name
    image_title: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    megapixels: Optional[float] = None


class VisionExtraction(BaseModel):
    """
    Structured vision data extracted from image by vision model.
    """
    description: str  # Scene description (2-3 sentences)
    detected_text: str  # OCR content
    concepts: List[str] = Field(default_factory=list)  # Object/concept tags
    confidence_scores: Optional[Dict[str, float]] = None  # Optional confidence metrics


class ImageResult(BaseModel):
    """
    Complete vision processing result.

    Includes all extracted data and formatted full_text for note creation.
    """
    item_id: str
    vision_data: VisionExtraction
    full_text: str  # Formatted markdown
    processing_time_seconds: float
    model_used: str  # e.g., "qwen2.5-vl:7b"


class QueueItem(BaseModel):
    """
    Queue item with status tracking.

    Mirrors podcast/models.py QueueItem structure.
    """
    image: ImageItem
    status: str = Field(default="pending", description="pending | processing | completed | failed | review")
    added_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None
    attempts: int = 0
    error_message: Optional[str] = None
    result_note_uuid: Optional[str] = None
