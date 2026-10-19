"""
Zettlecast Chunking Module

Splits note text into overlapping chunks for embedding.

Podcast transcript lines ("[12.3s - 45.6s] speaker_0: ...") are kept whole
where possible; when a long speaker turn has to be split, every piece keeps
the "[start] speaker:" header so search hits always show who spoke and when.
"""

import re
from typing import List

from .config import settings
from .models import ChunkModel

TURN_RE = re.compile(r"^\[(?P<start>[\d.]+)s - [\d.]+s\] (?P<speaker>[^:]+): (?P<text>.*)$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def recursive_split(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    separators: List[str] = None,
) -> List[str]:
    """
    Recursively split text using multiple separators.

    Tries to split on larger semantic boundaries first (paragraphs),
    then falls back to smaller ones (sentences, words).
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    separators = separators or ["\n\n", "\n", ". ", " ", ""]

    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    current_separator = separators[0]
    remaining_separators = separators[1:]
    parts = text.split(current_separator) if current_separator else list(text)

    current_chunk = ""
    for part in parts:
        part_with_sep = part + current_separator if current_separator else part

        if len(current_chunk) + len(part_with_sep) > chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            if len(part_with_sep) > chunk_size and remaining_separators:
                chunks.extend(recursive_split(
                    part,
                    chunk_size=chunk_size,
                    chunk_overlap=0,  # No overlap in recursion
                    separators=remaining_separators,
                ))
                current_chunk = ""
            else:
                current_chunk = part_with_sep
        else:
            current_chunk += part_with_sep

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev_chunk, chunk in zip(chunks, chunks[1:]):
            overlapped.append(prev_chunk[-chunk_overlap:] + " " + chunk)
        chunks = overlapped

    return chunks


def _split_transcript(text: str, chunk_size: int) -> List[str]:
    """Group transcript turns into chunks, splitting long turns by sentence."""
    pieces: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = TURN_RE.match(line)
        if not match or len(line) <= chunk_size:
            pieces.append(line)
            continue
        header = f"[{match['start']}s] {match['speaker']}:"
        current = ""
        for sentence in SENTENCE_RE.split(match["text"]):
            if current and len(header) + len(current) + len(sentence) + 2 > chunk_size:
                pieces.append(f"{header} {current.strip()}")
                current = ""
            current += sentence + " "
        if current.strip():
            pieces.append(f"{header} {current.strip()}")

    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > chunk_size:
            chunks.append(current)
            current = ""
        current = f"{current}\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


# The pipeline's own reports in a transcript note: not content, so never searched or linked on
# (their fixed wording made every episode "match" every other)
UNINDEXED_SECTIONS = ("Automatic Corrections", "Needs Review")


def _drop_sections(text: str, titles: tuple) -> str:
    """Remove '## <title>' sections (heading to the next '## ') from markdown."""
    parts = re.split(r"(?m)^(?=## )", text)
    return "".join(p for p in parts if not any(p.startswith(f"## {t}") for t in titles))


def create_chunks(text: str, base_chunk_id: str) -> List[ChunkModel]:
    """
    Create ChunkModel instances from text.

    Markdown sections before a "## Transcript" heading (summary, key points,
    chapters, show notes) are chunked as prose, without the verification reports;
    the transcript is chunked by speaker turn.
    """
    chunk_size = settings.chunk_size
    if "\n## Transcript\n" in text:
        head, transcript = text.split("\n## Transcript\n", 1)
        head = _drop_sections(head, UNINDEXED_SECTIONS)
        raw_chunks = recursive_split(head.strip()) + _split_transcript(transcript, chunk_size)
    else:
        raw_chunks = recursive_split(text)

    chunks = []
    current_pos = 0
    for i, chunk_text in enumerate(raw_chunks):
        if len(chunk_text.strip()) < settings.min_chunk_size and len(raw_chunks) > 1:
            continue
        start_char = text.find(chunk_text[:50], current_pos)
        if start_char == -1:
            start_char = current_pos
        current_pos = start_char
        chunks.append(ChunkModel(
            chunk_id=f"{base_chunk_id}_chunk_{i}",
            text=chunk_text,
            start_char=start_char,
            end_char=start_char + len(chunk_text),
        ))
    return chunks


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (4 chars per token on average)."""
    return len(text) // 4
