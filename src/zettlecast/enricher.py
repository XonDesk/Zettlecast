"""
Content Enricher Module

LLM-generated tags and a one-sentence summary for notes that don't already
have them (PDFs, web pages, markdown). Podcast transcripts get theirs from the
podcast enhancer and images from the vision step.
"""

import logging

from .config import settings
from .llm import LLMError, chat
from .models import NoteModel

logger = logging.getLogger(__name__)

ENRICH_PROMPT = """Read this document and return:
- "tags": exactly {count} short lowercase topic keywords that best describe it
- "summary": one concise sentence (max 150 characters) stating its main point

Title: {title}

Document (may be truncated):
{text}"""

ENRICH_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["tags", "summary"],
}


def enrich_note(note: NoteModel) -> NoteModel:
    """
    Fill in note.metadata.tags and note.metadata.custom["summary"] if missing.

    Failures (Ollama down, bad output) leave the note unchanged.
    """
    if not settings.enable_auto_tagging:
        return note
    if note.metadata.tags and "summary" in note.metadata.custom:
        return note
    if len(note.full_text.strip()) < 100:
        return note

    try:
        result = chat(
            ENRICH_PROMPT.format(
                count=settings.auto_tag_count,
                title=note.title,
                text=note.full_text[:40_000],
            ),
            schema=ENRICH_SCHEMA,
        )
    except LLMError as e:
        logger.warning(f"Enrichment skipped: {e}")
        return note

    if not note.metadata.tags:
        tags = [t.strip().lower() for t in result.get("tags", []) if isinstance(t, str) and t.strip()]
        note.metadata.tags = tags[: settings.auto_tag_count]
    if "summary" not in note.metadata.custom and result.get("summary"):
        note.metadata.custom["summary"] = result["summary"].strip()

    logger.info(f"Enriched '{note.title}': {note.metadata.tags}")
    return note
