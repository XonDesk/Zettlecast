"""
Content Enricher Module

LLM-generated topic tags, people, organisations and a one-sentence summary for notes
that don't already have them (PDFs, web pages, markdown). Podcast transcripts get theirs from the
podcast enhancer and images from the vision step.
"""

import logging

from .chunker import UNINDEXED_SECTIONS, _drop_sections
from .config import settings
from .llm import LLMError, chat
from .models import NoteModel

logger = logging.getLogger(__name__)

MIN_TEXT = 100  # shorter notes (a link or a title alone) give the LLM nothing to summarise

ENRICH_PROMPT = """Read this document and return:
- "tags": exactly {count} short lowercase topic keywords that best describe it: subjects,
  concepts and methods in their most common general form. No names of people or organisations.
- "people": full names of the people it is about or quotes in some depth (not passing mentions)
- "organisations": named organisations, institutions, companies, products, races and events
  it discusses in some depth
- "summary": one concise sentence (max 150 characters) stating its main point

Title: {title}

Document (may be truncated):
{text}"""

ENRICH_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"}},
        "people": {"type": "array", "items": {"type": "string"}},
        "organisations": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["tags", "people", "organisations", "summary"],
}


def _add_names(note: NoteModel, result: dict) -> None:
    """People (full names only) and organisations for the knowledge graph's entity links."""
    for field in ("people", "organisations"):
        if note.metadata.custom.get(field):
            continue
        names = [n.strip() for n in result.get(field) or [] if isinstance(n, str) and n.strip()]
        if field == "people":
            names = [n for n in names if " " in n]  # a first name alone can't tell people apart
        if names:
            note.metadata.custom[field] = list(dict.fromkeys(names))


def too_short(text: str) -> bool:
    return len(text.strip()) < MIN_TEXT


def enrich_note(note: NoteModel, raise_errors: bool = False) -> NoteModel:
    """
    Fill in note.metadata.tags and note.metadata.custom["summary"] if missing.

    Failures (LLM down, bad output) leave the note unchanged, or raise LLMError with raise_errors.
    An answer without a summary stores an empty one: the LLM had nothing to say, don't ask again.
    """
    if not settings.enable_auto_tagging or settings.is_primary:
        return note  # the primary has no LLM: the worker tags it later (tagging.py)
    if note.metadata.tags and "summary" in note.metadata.custom:
        return note
    if too_short(note.full_text):
        return note

    try:
        result = chat(
            ENRICH_PROMPT.format(
                count=settings.auto_tag_count,
                title=note.title,
                text=_drop_sections(note.full_text, UNINDEXED_SECTIONS)[:40_000],  # what's indexed
            ),
            schema=ENRICH_SCHEMA,
        )
    except LLMError as e:
        if raise_errors:
            raise
        logger.warning(f"Enrichment skipped: {e}")
        return note

    if not note.metadata.tags:
        tags = [t.strip().lower() for t in result.get("tags", []) if isinstance(t, str) and t.strip()]
        note.metadata.tags = tags[: settings.auto_tag_count]
    if "summary" not in note.metadata.custom:
        note.metadata.custom["summary"] = (result.get("summary") or "").strip()
    _add_names(note, result)

    logger.info(f"Enriched '{note.title}': {note.metadata.tags}")
    return note
