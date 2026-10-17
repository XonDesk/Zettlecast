"""
Zettlecast Output Formatter

Writes a transcription result as a markdown note with YAML frontmatter.
The note UUID is the episode ID, so reprocessing an episode replaces its note.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from ..config import settings
from .models import PodcastEpisode, TranscriptionResult


def format_transcript_for_zettlecast(
    result: TranscriptionResult,
    episode: PodcastEpisode,
    enhanced: Optional[dict] = None,
) -> str:
    """Format a transcription result as markdown with frontmatter."""
    enhanced = enhanced or {}
    transcript = enhanced.get("cleaned_transcript", result.full_text)
    keywords: List[str] = enhanced.get("keywords", result.keywords)
    sections: List[dict] = enhanced.get("sections", result.sections)
    summary: str = enhanced.get("summary", result.summary)
    key_points: List[str] = enhanced.get("key_points", result.key_points)
    speakers: dict = enhanced.get("speakers", {})
    uncertain: List[dict] = enhanced.get("uncertain_corrections", [])

    frontmatter = {
        "uuid": episode.id,
        "title": episode.episode_title or f"Podcast - {episode.podcast_name}",
        "source_type": "audio",
        "source": episode.audio_path,
        "status": "inbox",
        "created": datetime.utcnow().isoformat(),
        "duration_seconds": int(result.duration_seconds),
        "language": result.language,
        "speakers": result.speakers_detected,
    }
    if speakers:
        frontmatter["speaker_names"] = speakers
    if enhanced.get("verified_speakers"):
        # Named by the publisher's own transcript
        frontmatter["verified_speakers"] = enhanced["verified_speakers"]
    if keywords:
        frontmatter["tags"] = keywords[:10]
    if summary:
        frontmatter["summary"] = summary
    podcast_meta = {
        k: v
        for k, v in {
            "show": episode.podcast_name,
            "episode": episode.episode_title,
            "feed_url": episode.feed_url,
            "published": episode.published,
        }.items()
        if v
    }
    if podcast_meta:
        frontmatter["podcast"] = podcast_meta

    parts = [
        "---",
        yaml.dump(frontmatter, default_flow_style=False, sort_keys=False, allow_unicode=True).strip(),
        "---",
    ]
    if summary:
        parts += ["", "## Summary", "", summary]
    if key_points:
        parts += ["", "## Key Points", ""] + [f"- {p}" for p in key_points]
    if sections:
        parts += ["", "## Chapters", ""]
        for sec in sections:
            start = float(sec.get("start_time", 0) or 0)
            parts.append(f"- **{sec.get('name', 'Section')}** ([{start:.0f}s]): {sec.get('description', '')}")
    if episode.description:
        parts += ["", "## Show Notes", "", episode.description]
    if uncertain:
        parts += [
            "",
            "## Needs Review",
            "",
            "The following corrections were made with low confidence and should be verified:",
            "",
        ] + [f'- "{item.get("text", "")}"' for item in uncertain]
    parts += ["", "## Transcript", "", transcript, ""]
    return "\n".join(parts)


def transcript_filename(episode: PodcastEpisode) -> str:
    title = episode.episode_title or "podcast"
    safe_title = re.sub(r"[^\w\s-]", "", title)[:60].strip().replace(" ", "_")
    return f"{safe_title}_{episode.id[:8]}.md"


def save_result(
    result: TranscriptionResult,
    episode: PodcastEpisode,
    enhanced: Optional[dict] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Write the transcript note and return its path."""
    output_dir = Path(output_dir or settings.transcripts_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / transcript_filename(episode)
    file_path.write_text(format_transcript_for_zettlecast(result, episode, enhanced), encoding="utf-8")
    return file_path
