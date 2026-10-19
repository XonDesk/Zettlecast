"""
Zettlecast Data Models
Pydantic models for notes, chunks, and database schema.
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ChunkModel(BaseModel):
    """A chunk of text from a document, for vector storage."""

    chunk_id: str = Field(description="UUID of parent note + chunk index")
    text: str = Field(description="Chunk content")
    start_char: int = Field(description="Start position in original document")
    end_char: int = Field(description="End position in original document")


class NoteMetadata(BaseModel):
    """Flexible metadata for notes."""

    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    source_url: Optional[str] = None
    language: Optional[str] = None
    word_count: Optional[int] = None
    page_count: Optional[int] = None  # For PDFs
    duration_seconds: Optional[int] = None  # For audio
    embedded_media: List[str] = Field(default_factory=list)  # YouTube, video URLs
    custom: dict = Field(default_factory=dict)


class NoteModel(BaseModel):
    """Primary data model for a note in the knowledge base."""

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source_type: str = Field(description="pdf | web | audio | markdown | rss | image")
    source_path: str = Field(description="Original file path or URL")
    full_text: str = Field(description="Complete extracted text")
    metadata: NoteMetadata = Field(default_factory=NoteMetadata)
    content_hash: str = Field(description="SHA256 of full_text for dedup")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="inbox", description="inbox | reviewed | archived")



class ProcessingResult(BaseModel):
    """Result of processing a document."""

    status: str = Field(description="success | partial | failed")
    uuid: Optional[str] = None
    title: Optional[str] = None
    error_type: Optional[str] = None  # parse | embed | store | network | timeout
    error_message: Optional[str] = None
    retry_count: int = 0
    note: Optional[NoteModel] = None


class SearchResult(BaseModel):
    """A single search result."""

    uuid: str
    title: str
    score: float
    snippet: str = Field(description="Best-matching chunk of the note")
    source_type: str
    timestamp_seconds: Optional[float] = Field(
        default=None, description="Start of the matching transcript turn (audio notes)"
    )


class LinkSuggestion(BaseModel):
    """A suggested link between notes."""

    uuid: str
    title: str
    score: float
    reason: Optional[str] = Field(default=None, description="Why this note is related")
    kind: str = "similar"  # similar | duplicate | link
    passage: Optional[dict] = Field(
        default=None, description="Most alike passages: {here: {text, time}, there: {text, time}}"
    )
