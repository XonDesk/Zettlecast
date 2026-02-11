"""
Podcast Transcription Data Models

Pydantic models for podcast episodes, transcription results, and queue state.
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class PodcastEpisode(BaseModel):
    """Represents a podcast episode in the processing queue."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    audio_path: str = Field(description="Path to audio file")
    audio_hash: str = Field(description="SHA256 hash of audio file for deduplication")
    podcast_name: Optional[str] = Field(None, description="Show/podcast name")
    episode_title: Optional[str] = Field(None, description="Episode title")
    feed_url: Optional[str] = Field(None, description="RSS feed URL if from podcast feed")
    duration_seconds: Optional[int] = Field(None, description="Duration if known")


class TranscriptSegment(BaseModel):
    """A timestamped segment of transcribed speech with optional speaker label."""

    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text")
    speaker: Optional[str] = Field(None, description="Speaker label (e.g., 'Speaker 1')")


class TranscriptionResult(BaseModel):
    """Complete transcription output for an episode."""

    episode_id: str = Field(description="ID of the processed episode")
    segments: List[TranscriptSegment] = Field(default_factory=list)
    full_text: str = Field(description="Complete transcript as formatted string")
    language: str = Field(default="en", description="Detected language code")
    speakers_detected: int = Field(default=1, description="Number of unique speakers")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    sections: List[dict] = Field(default_factory=list, description="Detected sections/chapters")
    summary: str = Field(default="", description="Episode summary")
    key_points: List[str] = Field(default_factory=list, description="Key takeaways/insights")
    duration_seconds: float = Field(description="Audio duration")
    processing_time_seconds: float = Field(description="Time taken to process")


class QueueItem(BaseModel):
    """An item in the transcription processing queue."""

    episode: PodcastEpisode
    status: str = Field(
        default="pending",
        description="pending | processing | completed | failed | review",
    )
    attempts: int = Field(default=0, description="Number of processing attempts")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    added_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="When processing started")
    completed_at: Optional[datetime] = Field(None, description="When processing finished")
    result_path: Optional[str] = Field(None, description="Path to result JSON if completed")
