"""
Podcast RSS Feed Handler

Fetches podcast feeds and downloads episodes for transcription.
"""

import logging
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
import re

import xml.etree.ElementTree as ET

import feedparser
import httpx
from pydantic import BaseModel


logger = logging.getLogger(__name__)

USER_AGENT = "Zettlecast/0.1 (personal knowledge base; +https://github.com/XonDesk/Zettlecast)"
PODCAST_NS = "{https://podcastindex.org/namespace/1.0}"
# Preferred published-transcript formats: JSON carries per-word speakers and times
TRANSCRIPT_TYPES = ("application/json", "text/vtt", "application/x-subrip", "application/srt")


class FeedEpisode(BaseModel):
    """Represents a podcast episode from an RSS feed."""
    title: str
    audio_url: str
    published: str
    duration: Optional[str] = None
    description: Optional[str] = None
    guid: Optional[str] = None
    transcript_url: Optional[str] = None


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


def episode_date(published: str) -> str:
    """Feed publish date (RFC 822) as YYYY-MM-DD, for telling same-titled episodes apart."""
    try:
        return parsedate_to_datetime(published).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return sanitize_filename(published) or "undated"


def html_to_text(html: str) -> str:
    """Show notes HTML -> plain text with one paragraph/list item per line."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "li", "div", "h1", "h2", "h3", "h4", "h5", "h6", "tr"]):
        block.append("\n")
    text = soup.get_text()
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def transcript_urls(feed_xml: bytes) -> dict:
    """
    guid -> best <podcast:transcript> URL per item. feedparser keeps only one
    transcript tag per item, so read them from the XML directly.
    """
    try:
        root = ET.fromstring(feed_xml)
    except ET.ParseError as e:
        logger.warning(f"Could not read transcript links from feed: {e}")
        return {}
    urls = {}
    for item in root.iter("item"):
        guid = (item.findtext("guid") or "").strip()
        tags = {t.get("type"): t.get("url") for t in item.findall(f"{PODCAST_NS}transcript")}
        best = next((tags[t] for t in TRANSCRIPT_TYPES if tags.get(t)), None)
        if guid and best:
            urls[guid] = best
    return urls


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
    
    response = httpx.get(feed_url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60)
    response.raise_for_status()
    feed = feedparser.parse(response.content)
    transcripts = transcript_urls(response.content)

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

        # content:encoded holds the full show notes; description may be truncated
        content = entry.get("content") or []
        notes_html = content[0].get("value", "") if content else entry.get("description", "")

        episodes.append(FeedEpisode(
            title=entry.get("title", "Untitled Episode"),
            audio_url=audio_url,
            published=entry.get("published", ""),
            duration=entry.get("itunes_duration", ""),
            description=html_to_text(notes_html),
            guid=entry.get("id", audio_url),
            transcript_url=transcripts.get(entry.get("id", "")),
        ))
        
    return PodcastFeed(
        title=feed.feed.get("title", "Unknown Podcast"),
        description=feed.feed.get("description", ""),
        episodes=episodes
    )


def download_episode(
    episode: FeedEpisode, 
    output_dir: Path,
    client: Optional[httpx.Client] = None,
    stem: Optional[str] = None,
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
    safe_title = stem or sanitize_filename(episode.title)
    filename = f"{safe_title}{ext}"
    output_path = output_dir / filename
    
    if output_path.exists():
        logger.info(f"File already exists: {output_path.name}")
        return output_path
        
    logger.info(f"Downloading: {episode.title}")
    
    # Use a proper User-Agent to avoid 403 errors from podcast hosts
    headers = {"User-Agent": USER_AGENT}
    
    try:
        if client:
            response = client.get(episode.audio_url, follow_redirects=True, headers=headers)
            response.raise_for_status()
            content = response.content
        else:
            with httpx.Client(timeout=120.0) as c:
                response = c.get(episode.audio_url, follow_redirects=True, headers=headers)
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
