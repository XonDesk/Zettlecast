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
    """
    try:
        import trafilatura
        
        # Download page
        logger.info(f"Fetching URL: {url}")
        downloaded = trafilatura.fetch_url(url)
        
        if not downloaded:
            return ProcessingResult(
                status="failed",
                error_type="network",
                error_message="Failed to download URL",
            )
        
        # Extract main content
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            output_format="markdown",
        )
        
        if not text:
            return ProcessingResult(
                status="failed",
                error_type="parse",
                error_message="No content extracted from URL",
            )
        
        # Extract metadata
        metadata_dict = trafilatura.extract_metadata(downloaded)
        title = metadata_dict.title if metadata_dict else urlparse(url).netloc
        
        uuid = generate_uuid()
        
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
            ),
        )
        
        return ProcessingResult(
            status="success",
            uuid=uuid,
            title=title,
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
        return parse_pdf(file_path)
    elif suffix in {".md", ".markdown", ".txt"}:
        return parse_markdown(file_path)
    elif suffix in {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm"}:
        return parse_audio(file_path)
    else:
        return ProcessingResult(
            status="failed",
            error_type="parse",
            error_message=f"Unsupported file type: {suffix}",
        )
