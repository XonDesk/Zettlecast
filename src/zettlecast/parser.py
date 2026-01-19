"""
Zettlecast Parser Module
Tiered PDF parsing, web scraping, and audio transcription.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx

from .config import settings
from .enricher import enrich_note
from .identity import compute_content_hash, generate_uuid
from .models import NoteMetadata, NoteModel, ProcessingResult

logger = logging.getLogger(__name__)


# --- PDF Parsing (Tiered) ---

def extract_pdf_pypdf(file_path: Path) -> Tuple[str, int]:
    """
    Extract text from PDF using pypdf (fast, basic).
    
    Returns:
        Tuple of (extracted_text, page_count)
    """
    from pypdf import PdfReader
    
    reader = PdfReader(str(file_path))
    pages = []
    
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    
    return "\n\n".join(pages), len(reader.pages)


def extract_pdf_marker(file_path: Path) -> Tuple[str, int]:
    """
    Extract text from PDF using Marker (quality, handles complex layouts).
    
    Returns:
        Tuple of (extracted_markdown, page_count)
    """
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    
    # Load models (cached after first call)
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    
    # Convert
    result = converter(str(file_path))
    
    return result.markdown, result.metadata.get("page_count", 0)


def parse_pdf(file_path: Path) -> ProcessingResult:
    """
    Parse PDF with tiered fallback strategy.
    
    1. Try pypdf (fast)
    2. If extraction ratio < 50%, try Marker
    """
    try:
        # Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > settings.max_file_size_mb:
            return ProcessingResult(
                status="failed",
                error_type="parse",
                error_message=f"File too large: {size_mb:.1f}MB > {settings.max_file_size_mb}MB limit",
            )
        
        # Tier 1: pypdf
        logger.info(f"Parsing PDF with pypdf: {file_path.name}")
        text, page_count = extract_pdf_pypdf(file_path)
        
        # Check extraction quality (rough heuristic)
        chars_per_page = len(text) / max(page_count, 1)
        
        if chars_per_page < 500 and settings.use_marker_fallback:
            # Low extraction, try Marker
            logger.info(f"Low extraction ({chars_per_page:.0f} chars/page), trying Marker")
            try:
                text, page_count = extract_pdf_marker(file_path)
            except Exception as e:
                logger.warning(f"Marker fallback failed: {e}, using pypdf result")
        
        if not text.strip():
            return ProcessingResult(
                status="failed",
                error_type="parse",
                error_message="No text extracted from PDF",
            )
        
        # Create note
        uuid = generate_uuid()
        title = file_path.stem.replace("_", " ").replace("-", " ")
        
        note = NoteModel(
            uuid=uuid,
            title=title,
            source_type="pdf",
            source_path=str(file_path),
            full_text=text,
            content_hash=compute_content_hash(text),
            metadata=NoteMetadata(page_count=page_count),
        )
        
        return ProcessingResult(
            status="success",
            uuid=uuid,
            title=title,
            note=note,
        )
    
    except Exception as e:
        logger.exception(f"PDF parsing failed: {e}")
        return ProcessingResult(
            status="failed",
            error_type="parse",
            error_message=str(e),
        )


# --- Web Scraping ---

def parse_url(url: str) -> ProcessingResult:
    """
    Extract main content from a URL using trafilatura.
    Enhanced to extract richer metadata and embedded media.
    """
    try:
        import re
        import trafilatura
        from bs4 import BeautifulSoup
        
        # Browser-like headers to bypass bot detection
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }
        
        # Download page with custom headers
        logger.info(f"Fetching URL: {url}")
        try:
            response = httpx.get(url, headers=headers, follow_redirects=True, timeout=30)
            response.raise_for_status()
            downloaded = response.text
        except httpx.HTTPStatusError as e:
            return ProcessingResult(
                status="failed",
                error_type="network",
                error_message=f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
            )
        except httpx.RequestError as e:
            return ProcessingResult(
                status="failed",
                error_type="network",
                error_message=f"Request failed: {str(e)}",
            )
        
        if not downloaded:
            return ProcessingResult(
                status="failed",
                error_type="network",
                error_message="Failed to download URL",
            )
        
        # Extract main content with links preserved
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            include_links=True,  # Preserve hyperlinks in markdown
            output_format="markdown",
        )
        
        # Fallback to BeautifulSoup if trafilatura extraction fails
        if not text:
            logger.warning(f"Trafilatura extraction failed for {url} (content length: {len(downloaded)}), attempting fallback.")
            soup = BeautifulSoup(downloaded, "html.parser")
            
            # Remove scripts and styles
            for script in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator="\n\n")
            
            # Clean up empty lines
            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(line for line in lines if line)
        
        if not text:
            return ProcessingResult(
                status="failed",
                error_type="parse",
                error_message="No content extracted from URL (tried trafilatura and basic fallback)",
            )
        
        # Extract metadata
        metadata_dict = trafilatura.extract_metadata(downloaded)
        title = metadata_dict.title if metadata_dict else urlparse(url).netloc
        
        # Extract embedded media (YouTube, Vimeo iframes)
        embedded_media = []
        try:
            soup = BeautifulSoup(downloaded, "html.parser")
            
            # Find YouTube iframes
            for iframe in soup.find_all("iframe"):
                src = iframe.get("src", "")
                if "youtube.com" in src or "youtu.be" in src or "vimeo.com" in src:
                    embedded_media.append(src)
            
            # Also check for youtube-nocookie.com embeds
            for iframe in soup.find_all("iframe"):
                src = iframe.get("src", "")
                if "youtube-nocookie.com" in src:
                    embedded_media.append(src)
                    
            # Find video tags
            for video in soup.find_all("video"):
                src = video.get("src")
                if src:
                    embedded_media.append(src)
                # Check source tags inside video
                for source in video.find_all("source"):
                    src = source.get("src")
                    if src:
                        embedded_media.append(src)
                        
        except Exception as e:
            logger.warning(f"Failed to extract embedded media: {e}")
        
        # Calculate word count
        word_count = len(text.split()) if text else None
        
        # Extract tags/categories
        tags = []
        if metadata_dict:
            if hasattr(metadata_dict, "categories") and metadata_dict.categories:
                tags.extend(metadata_dict.categories)
            if hasattr(metadata_dict, "tags") and metadata_dict.tags:
                tags.extend(metadata_dict.tags)
        
        # Extract language
        language = None
        if metadata_dict and hasattr(metadata_dict, "language"):
            language = metadata_dict.language
        
        uuid = generate_uuid()
        
        # Add embedded media info to text if found
        if embedded_media:
            text += "\n\n---\n**Embedded Media:**\n"
            for media_url in embedded_media:
                text += f"- {media_url}\n"
        
        note = NoteModel(
            uuid=uuid,
            title=title or "Untitled",
            source_type="web",
            source_path=url,
            full_text=text,
            content_hash=compute_content_hash(text),
            metadata=NoteMetadata(
                source_url=url,
                author=metadata_dict.author if metadata_dict else None,
                tags=tags,
                language=language,
                word_count=word_count,
                embedded_media=embedded_media,
            ),
        )
        
        # Enrich with LLM-generated tags and summary
        note = enrich_note(note)
        
        return ProcessingResult(
            status="success",
            uuid=uuid,
            title=title,
            note=note,
        )
    
    except Exception as e:
        logger.exception(f"URL parsing failed: {e}")
        return ProcessingResult(
            status="failed",
            error_type="parse",
            error_message=str(e),
        )


# --- Audio Transcription ---

def parse_audio(file_path: Path) -> ProcessingResult:
    """
    Transcribe audio file using faster-whisper.
    """
    try:
        from faster_whisper import WhisperModel
        
        # Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > settings.max_file_size_mb * 2:  # Allow larger for audio
            return ProcessingResult(
                status="failed",
                error_type="parse",
                error_message=f"Audio file too large: {size_mb:.1f}MB",
            )
        
        logger.info(f"Transcribing audio: {file_path.name} (model: {settings.whisper_model})")
        
        # Load model
        model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device if settings.whisper_device != "auto" else "auto",
            compute_type="int8",
        )
        
        # Transcribe
        segments, info = model.transcribe(str(file_path), beam_size=5)
        
        # Build transcript with timestamps
        lines = []
        for segment in segments:
            timestamp = f"[{segment.start:.1f}s]"
            lines.append(f"{timestamp} {segment.text.strip()}")
        
        text = "\n".join(lines)
        
        if not text.strip():
            return ProcessingResult(
                status="failed",
                error_type="parse",
                error_message="No speech detected in audio",
            )
        
        uuid = generate_uuid()
        title = file_path.stem.replace("_", " ").replace("-", " ")
        duration = int(info.duration) if info.duration else None
        
        note = NoteModel(
            uuid=uuid,
            title=title,
            source_type="audio",
            source_path=str(file_path),
            full_text=text,
            content_hash=compute_content_hash(text),
            metadata=NoteMetadata(
                duration_seconds=duration,
                language=info.language if hasattr(info, "language") else None,
            ),
        )
        
        return ProcessingResult(
            status="success",
            uuid=uuid,
            title=title,
            note=note,
        )
    
    except Exception as e:
        logger.exception(f"Audio transcription failed: {e}")
        return ProcessingResult(
            status="failed",
            error_type="parse",
            error_message=str(e),
        )


# --- RSS Parsing ---

def parse_rss_feed(feed_url: str) -> list[ProcessingResult]:
    """
    Parse RSS feed and extract individual articles.
    
    Returns a list of ProcessingResult for each article.
    """
    try:
        import feedparser
        
        logger.info(f"Parsing RSS feed: {feed_url}")
        feed = feedparser.parse(feed_url)
        
        if feed.bozo:
            return [ProcessingResult(
                status="failed",
                error_type="parse",
                error_message=f"Invalid RSS feed: {feed.bozo_exception}",
            )]
        
        results = []
        for entry in feed.entries[:10]:  # Limit to 10 entries
            # Extract full content via trafilatura if link available
            if hasattr(entry, "link") and entry.link:
                result = parse_url(entry.link)
                if result.status == "success":
                    result.title = entry.get("title", result.title)
                results.append(result)
        
        return results
    
    except Exception as e:
        logger.exception(f"RSS parsing failed: {e}")
        return [ProcessingResult(
            status="failed",
            error_type="parse",
            error_message=str(e),
        )]


# --- Markdown Parsing ---

def parse_markdown(file_path: Path) -> ProcessingResult:
    """
    Parse a local markdown file, ensuring UUID in frontmatter.
    """
    try:
        from .identity import ensure_uuid_in_file, parse_frontmatter
        
        content = file_path.read_text(encoding="utf-8")
        uuid, was_modified = ensure_uuid_in_file(file_path)
        
        if was_modified:
            content = file_path.read_text(encoding="utf-8")
        
        frontmatter, body = parse_frontmatter(content)
        title = frontmatter.get("title", file_path.stem)
        
        note = NoteModel(
            uuid=uuid,
            title=title,
            source_type="markdown",
            source_path=str(file_path),
            full_text=body,
            content_hash=compute_content_hash(body),
            metadata=NoteMetadata(
                tags=frontmatter.get("tags", []),
            ),
        )
        
        return ProcessingResult(
            status="success",
            uuid=uuid,
            title=title,
            note=note,
        )
    
    except Exception as e:
        logger.exception(f"Markdown parsing failed: {e}")
        return ProcessingResult(
            status="failed",
            error_type="parse",
            error_message=str(e),
        )


# --- Unified Parser ---

def parse_file(file_path: Path) -> ProcessingResult:
    """
    Parse any supported file type.
    
    Dispatches to appropriate parser based on file extension.
    """
    suffix = file_path.suffix.lower()
    
    if suffix == ".pdf":
        result = parse_pdf(file_path)
    elif suffix in {".md", ".markdown", ".txt"}:
        result = parse_markdown(file_path)
    elif suffix in {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm"}:
        result = parse_audio(file_path)
    else:
        return ProcessingResult(
            status="failed",
            error_type="parse",
            error_message=f"Unsupported file type: {suffix}",
        )
    
    # Enrich successful parses with LLM-generated tags and summary
    if result.status == "success" and result.note:
        result.note = enrich_note(result.note)
    
    return result
