"""
Zettlecast Output Formatter

Formats transcription results into Zettlecast-compatible markdown
with proper YAML frontmatter.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import yaml

from ..config import settings
from .models import PodcastEpisode, TranscriptionResult


def format_transcript_for_zettlecast(
    result: TranscriptionResult,
    episode: PodcastEpisode,
    enhanced: Optional[dict] = None,
) -> str:
    """
    Format transcription result as Zettlecast-compatible markdown.

    Args:
        result: Transcription result
        episode: Episode metadata
        enhanced: Optional dict with cleaned_transcript, keywords, sections

    Returns:
        Complete markdown content with frontmatter
    """
    # Use enhanced transcript if available
    transcript = (
        enhanced.get("cleaned_transcript", result.full_text)
        if enhanced
        else result.full_text
    )
    keywords = enhanced.get("keywords", result.keywords) if enhanced else result.keywords
    sections = enhanced.get("sections", result.sections) if enhanced else result.sections

    # Build frontmatter
    frontmatter = {
        "uuid": str(uuid4()),
        "title": episode.episode_title or f"Podcast - {episode.podcast_name}",
        "source_type": "audio",
        "source": episode.audio_path,
        "status": "inbox",
        "created": datetime.utcnow().isoformat(),
        "duration_seconds": int(result.duration_seconds),
        "language": result.language,
        "speakers": result.speakers_detected,
    }

    # Add tags from keywords
    if keywords:
        frontmatter["tags"] = keywords[:10]  # Limit to 10 tags

    # Add podcast metadata
    podcast_meta = {}
    if episode.podcast_name:
        podcast_meta["show"] = episode.podcast_name
    if episode.episode_title:
        podcast_meta["episode"] = episode.episode_title
    if episode.feed_url:
        podcast_meta["feed_url"] = episode.feed_url
    if podcast_meta:
        frontmatter["podcast"] = podcast_meta

    # Build header comments (metadata hints for Zettlecast)
    header_lines = [
        f"# Language: {result.language}",
        f"# Speakers: {result.speakers_detected}",
        f"# Duration: {int(result.duration_seconds)}s",
    ]
    if keywords:
        header_lines.append(f"# Keywords: {', '.join(keywords[:5])}")

    header = "\n".join(header_lines)

    # Add sections as markdown headers if detected
    sections_md = ""
    if sections:
        sections_md = "\n\n## Chapters\n\n"
        for sec in sections:
            name = sec.get("name", "Section")
            start = sec.get("start_time", 0)
            desc = sec.get("description", "")
            sections_md += f"- **{name}** ([{start:.0f}s]): {desc}\n"

    # Serialize frontmatter
    frontmatter_yaml = yaml.dump(
        frontmatter,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )

    # Combine all parts
    content = f"""---
{frontmatter_yaml.strip()}
---

{header}
{sections_md}

## Transcript

{transcript}
"""

    return content


def save_result(
    result: TranscriptionResult,
    episode: PodcastEpisode,
    enhanced: Optional[dict] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save transcription result as markdown file.

    Args:
        result: Transcription result
        episode: Episode metadata
        enhanced: Enhanced data (cleaned transcript, keywords, sections)
        output_dir: Output directory (defaults to STORAGE_PATH)

    Returns:
        Path to saved file
    """
    output_dir = output_dir or settings.storage_path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Format content
    content = format_transcript_for_zettlecast(result, episode, enhanced)

    # Create safe filename
    title = episode.episode_title or "podcast"
    safe_title = re.sub(r"[^\w\s-]", "", title)[:50].strip().replace(" ", "_")
    uuid_prefix = result.episode_id[:8]
    filename = f"{safe_title}_{uuid_prefix}.md"

    file_path = output_dir / filename
    file_path.write_text(content, encoding="utf-8")

    return file_path


def save_result_json(
    result: TranscriptionResult,
    episode: PodcastEpisode,
    enhanced: Optional[dict] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save transcription result as JSON file.

    Useful for programmatic access or debugging.

    Args:
        result: Transcription result
        episode: Episode metadata
        enhanced: Enhanced data
        output_dir: Output directory

    Returns:
        Path to saved JSON file
    """
    import json

    output_dir = output_dir or settings.storage_path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build JSON structure
    data = {
        "episode": episode.model_dump(),
        "result": result.model_dump(),
        "enhanced": enhanced or {},
    }

    # Create filename
    title = episode.episode_title or "podcast"
    safe_title = re.sub(r"[^\w\s-]", "", title)[:50].strip().replace(" ", "_")
    uuid_prefix = result.episode_id[:8]
    filename = f"{safe_title}_{uuid_prefix}.json"

    file_path = output_dir / filename
    file_path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )

    return file_path
