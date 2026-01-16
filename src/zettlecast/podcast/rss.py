"""
Podcast RSS Feed Handler

Fetches podcast feeds and downloads episodes for transcription.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse
import re

import feedparser
import httpx
from pydantic import BaseModel

from ..config import settings

logger = logging.getLogger(__name__)


class FeedEpisode(BaseModel):
    """Represents a podcast episode from an RSS feed."""
    title: str
    audio_url: str
    published: str
    duration: Optional[str] = None
    description: Optional[str] = None
    guid: Optional[str] = None


class PodcastFeed(BaseModel):
    """Represents a parsed podcast feed."""
    title: str
    description: Optional[str] = None
    episodes: List[FeedEpisode] = []


def sanitize_filename(name: str) -> str:
    """Sanitize string for use as filename."""
    # Remove invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Truncate
    return name[:200]


def parse_feed(feed_url: str, limit: int = 5) -> PodcastFeed:
    """
    Parse a podcast RSS feed.
    
    Args:
        feed_url: URL of the RSS feed
        limit: Max number of episodes to retrieve
        
    Returns:
        PodcastFeed object with metadata and episodes
    """
    logger.info(f"Parsing podcast feed: {feed_url}")
    
    feed = feedparser.parse(feed_url)
    
    if feed.bozo:
        logger.warning(f"Feed parsing warning: {feed.bozo_exception}")
        # Continue if possible, as feedparser often works despite errors
        
    if not feed.entries:
        raise ValueError("No episodes found in feed")

    episodes = []
    
    for entry in feed.entries[:limit]:
        # Find audio enclosure
        audio_url = None
        for link in entry.get("links", []):
            if link.get("type", "").startswith("audio/"):
                audio_url = link.get("href")
                break
        
        # Fallback to enclosures list
        if not audio_url and hasattr(entry, "enclosures"):
            for enc in entry.enclosures:
                if enc.get("type", "").startswith("audio/"):
                    audio_url = enc.get("href")
                    break
        
        if not audio_url:
            continue
            
        episodes.append(FeedEpisode(
            title=entry.get("title", "Untitled Episode"),
            audio_url=audio_url,
            published=entry.get("published", ""),
            duration=entry.get("itunes_duration", ""),
            description=entry.get("description", ""),
            guid=entry.get("id", audio_url)
        ))
        
    return PodcastFeed(
        title=feed.feed.get("title", "Unknown Podcast"),
        description=feed.feed.get("description", ""),
        episodes=episodes
    )


def download_episode(
    episode: FeedEpisode, 
    output_dir: Path,
    client: Optional[httpx.Client] = None
) -> Path:
    """
    Download podcast episode audio to local file.
    
    Args:
        episode: FeedEpisode object
        output_dir: Directory to save file
        client: Optional httpx Client
        
    Returns:
        Path to downloaded file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine extension from URL
    path = urlparse(episode.audio_url).path
    ext = Path(path).suffix or ".mp3"
    
    # Create safe filename
    safe_title = sanitize_filename(episode.title)
    filename = f"{safe_title}{ext}"
    output_path = output_dir / filename
    
    if output_path.exists():
        logger.info(f"File already exists: {output_path.name}")
        return output_path
        
    logger.info(f"Downloading: {episode.title}")
    
    try:
        if client:
            response = client.get(episode.audio_url, follow_redirects=True)
            response.raise_for_status()
            content = response.content
        else:
            with httpx.Client(timeout=30.0) as c:
                response = c.get(episode.audio_url, follow_redirects=True)
                response.raise_for_status()
                content = response.content
                
        output_path.write_bytes(content)
        return output_path
        
    except Exception as e:
        logger.error(f"Download failed for {episode.title}: {e}")
        # Clean up partial file if needed (though write_bytes is atomic-ish)
        if output_path.exists():
            output_path.unlink()
        raise
