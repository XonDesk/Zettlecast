"""
Zettlecast Parser Module
PDF parsing, web scraping, markdown and image ingestion.
"""

import logging
import re
from pathlib import Path
from typing import Tuple
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

IMAGE_MD = re.compile(r"!\[[^\]\n]*\]\([^)\s]+\)")
BYLINE = re.compile(r"(?im)^by:?\s+\**([^*\n]{3,60}?)\**\s*$")


def _heading_level(line: str) -> int:
    match = re.match(r"(#{1,6}) ", line)
    return len(match.group(1)) if match else 0


def tidy_web_markdown(text: str) -> str:
    """Images on lines of their own (trafilatura glues them to the next paragraph), and no
    headings left empty by what it drops (a table of contents made of links)."""
    text = re.sub(rf"[ \t]*({IMAGE_MD.pattern})[ \t]*", r"\n\n\1\n\n", text)
    lines = [l.rstrip() for l in text.splitlines()]
    kept: list[str] = []
    for line in reversed(lines):  # from the end, so a section emptied this way empties its parent
        level = _heading_level(line)
        if level:
            nxt = next((l for l in reversed(kept) if l), "")
            if not nxt or 0 < _heading_level(nxt) <= level:
                continue  # nothing under it but the next section of the same or a higher level
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(reversed(kept))).strip()


def byline_author(text: str) -> str | None:
    """'By Carlo M. Cipolla' near the top, for pages whose metadata names no author."""
    match = BYLINE.search("\n".join(text.splitlines()[:12]))
    return match.group(1).strip() if match else None


def parse_url(url: str, enrich: bool = True) -> ProcessingResult:
    """
    Extract main content from a URL using trafilatura.
    Enhanced to extract richer metadata and embedded media.
    """
    try:
        import trafilatura
        from bs4 import BeautifulSoup
        
        # Sites disagree on what they block: Wikipedia rejects browser-looking
        # agents, others reject anything that isn't a browser. Try both.
        common = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        user_agents = [
            "Zettlecast/0.1 (personal knowledge base; +https://github.com/XonDesk/Zettlecast)",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        ]

        logger.info(f"Fetching URL: {url}")
        try:
            for user_agent in user_agents:
                response = httpx.get(
                    url,
                    headers={**common, "User-Agent": user_agent},
                    follow_redirects=True,
                    timeout=30,
                )
                if response.status_code not in (403, 429):
                    break
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
        
        # Extract main content with links and the article's figures preserved
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            include_links=True,  # Preserve hyperlinks in markdown
            include_images=True,
            output_format="markdown",
            url=url,  # image paths made absolute
        )
        if text:
            text = tidy_web_markdown(text)
        
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
        # Site categories/tags are often navigation junk ("See More Articles"):
        # keep them as metadata and let the LLM enricher choose topic tags.
        site_tags = []
        if metadata_dict:
            site_tags = list(getattr(metadata_dict, "categories", None) or []) + list(getattr(metadata_dict, "tags", None) or [])
        
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
                author=(metadata_dict.author if metadata_dict else None) or byline_author(text),
                custom={"site_tags": site_tags} if site_tags else {},
                language=language,
                word_count=word_count,
                embedded_media=embedded_media,
            ),
        )
        
        # Enrich with LLM-generated tags and summary (not for text read into another note)
        if enrich:
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


# --- Markdown Parsing ---

def parse_markdown(file_path: Path) -> ProcessingResult:
    """
    Parse a markdown note, ensuring it has a UUID in its frontmatter.

    Frontmatter written by the podcast pipeline (source_type, source, tags,
    summary, duration, created) is carried into the note, so transcripts
    are stored as audio notes that point at their episode file.
    """
    try:
        from datetime import datetime

        from .identity import ensure_uuid_in_file, parse_frontmatter

        uuid, _ = ensure_uuid_in_file(file_path)
        frontmatter, body = parse_frontmatter(file_path.read_text(encoding="utf-8"))
        title = str(frontmatter.get("title") or file_path.stem)

        custom = {"note_path": str(file_path)}
        for key in ("summary", "podcast", "speaker_names", "speakers", "people", "organisations"):
            if frontmatter.get(key):
                custom[key] = frontmatter[key]

        created = frontmatter.get("created")
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created)
            except ValueError:
                created = None

        tags = frontmatter.get("tags") or []
        note = NoteModel(
            uuid=uuid,
            title=title,
            source_type=frontmatter.get("source_type", "markdown"),
            source_path=str(frontmatter.get("source") or file_path),
            full_text=body,
            content_hash=compute_content_hash(body),
            status=frontmatter.get("status", "inbox"),
            metadata=NoteMetadata(
                tags=[str(t) for t in tags] if isinstance(tags, list) else [],
                duration_seconds=frontmatter.get("duration_seconds"),
                language=frontmatter.get("language"),
                word_count=len(body.split()),
                custom=custom,
            ),
            **({"created_at": created} if isinstance(created, datetime) else {}),
        )
        return ProcessingResult(status="success", uuid=uuid, title=title, note=note)

    except Exception as e:
        logger.exception(f"Markdown parsing failed: {e}")
        return ProcessingResult(status="failed", error_type="parse", error_message=str(e))


# --- Unified Parser ---

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm", ".aac", ".opus"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}


def parse_file(file_path: Path) -> ProcessingResult:
    """
    Parse a PDF, markdown/text or image file into a note.

    Audio is not handled here; it goes through the podcast queue
    (see pipeline.ingest_path) for transcription with speaker labels.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        result = parse_pdf(file_path)
    elif suffix in TEXT_EXTENSIONS:
        result = parse_markdown(file_path)
    elif suffix in IMAGE_EXTENSIONS:
        from .image.image_parser import parse_image
        result = parse_image(file_path)
    else:
        return ProcessingResult(
            status="failed",
            error_type="parse",
            error_message=f"Unsupported file type: {suffix}",
        )

    # Images get tags from the vision step and transcripts from the enhancer
    if result.status == "success" and result.note and result.note.source_type not in ("image", "audio"):
        result.note = enrich_note(result.note)

    return result
