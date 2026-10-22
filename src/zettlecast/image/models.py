"""
Image Processing Data Models

Data structures for image queue management and vision processing results.
"""

from datetime import datetime
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
    keep_image: bool = True  # False: only the extracted information is kept (screenshot triage "keep")
    # Part of the image the user circled in triage: {"box": [x, y, w, h], "points": [[x, y], ...]},
    # fractions of the image size
    focus_region: Optional[dict] = None


class VisionExtraction(BaseModel):
    """
    Structured vision data extracted from image by vision model.
    """
    # what the user kept it for (media | quote | idea | tip | place | product | recipe | reference |
    # other) and the thing itself: name, by, type, where, price (only the ones known)
    saved_as: str = "other"
    item: Dict[str, str] = Field(default_factory=dict)
    kind: str = "other"  # screenshot | photo | other
    focus: str = ""  # screenshot: the content it was taken for, word for word
    context: str = ""  # screenshot: app/site, author, what the focus replies to
    title: str = ""
    description: str  # Scene description (2-3 sentences)
    detected_text: str  # OCR content
    concepts: List[str] = Field(default_factory=list)  # Object/concept tags
    people: List[str] = Field(default_factory=list)  # Full names shown or named
    organisations: List[str] = Field(default_factory=list)
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
    worker_id: Optional[str] = None  # worker machine that claimed it
    lease_until: Optional[datetime] = None  # claim runs out then unless the worker renews it
