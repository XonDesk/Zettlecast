"""
Zettlecast Search & Retrieval

Query -> embed -> nearest chunks -> rerank chunks -> best chunk per note.
"""

import logging
import re
from typing import List, Optional

from .config import settings
from .db import db
from .embeddings import embed_query, rerank
from .models import LinkSuggestion, SearchResult

logger = logging.getLogger(__name__)

_TIMESTAMP_RE = re.compile(r"^\[([\d.]+)s")


def _chunk_timestamp(text: str) -> Optional[float]:
    match = _TIMESTAMP_RE.match(text.lstrip())
    return float(match.group(1)) if match else None


def search(
    query: str,
    top_k: int = None,
    rerank_results: bool = True,
) -> List[SearchResult]:
    """
    Semantic search over every chunk of every note.

    Returns one result per note (its best-matching chunk), best first.
    """
    top_k = top_k or settings.return_top_k

    candidates = db.search_chunks(embed_query(query), top_k=settings.rerank_top_k)
    if not candidates:
        return []

    if rerank_results:
        ranked = rerank(query, [c["text"] for c in candidates])
        scored = [(candidates[i], score) for i, score in ranked if score >= settings.search_min_score]
    else:
        scored = [(c, c["similarity"]) for c in candidates]

    # Keep the best chunk per note
    best: dict = {}
    for chunk, score in scored:
        if chunk["note_uuid"] not in best:
            best[chunk["note_uuid"]] = (chunk, score)
        if len(best) >= top_k:
            break

    titles = db.note_titles(list(best))
    results = []
    for uuid, (chunk, score) in best.items():
        title, source_type = titles.get(uuid, ("Untitled", "unknown"))
        results.append(SearchResult(
            uuid=uuid,
            title=title,
            score=round(float(score), 4),
            snippet=chunk["text"],
            source_type=source_type,
            timestamp_seconds=_chunk_timestamp(chunk["text"]),
        ))
    return results


def get_suggestions_for_note(note_uuid: str, limit: int = 5) -> List[LinkSuggestion]:
    """Related notes for the detail view: the note's graph edges, strongest first."""
    edges = db.get_edges_for_note(note_uuid)
    edges.sort(key=lambda e: e["weight"], reverse=True)
    others = [e["target_uuid"] if e["source_uuid"] == note_uuid else e["source_uuid"] for e in edges]
    titles = db.note_titles(others)
    return [
        LinkSuggestion(
            uuid=other,
            title=titles[other][0],
            score=round(float(edge["weight"]), 4),
            reason=f"similarity {edge['vector_sim']:.2f}, shared tags {edge['tag_sim']:.2f}",
        )
        for edge, other in zip(edges, others)
        if other in titles
    ][:limit]


def accept_link(source_uuid: str, target_uuid: str) -> bool:
    """
    Accept a suggested link by adding a [[wikilink]] to the source note's file.

    Returns:
        True if link was added, False if already present or no file to edit
    """
    from pathlib import Path

    from .identity import add_wikilink_to_file

    source_note = db.get_note_by_uuid(source_uuid)
    target_note = db.get_note_by_uuid(target_uuid)
    if not source_note or not target_note:
        return False

    path = Path(source_note.metadata.custom.get("note_path") or source_note.source_path)
    if path.suffix.lower() in {".md", ".markdown"} and path.exists():
        return add_wikilink_to_file(path, target_note.title)
    return False


def reject_link(source_uuid: str, target_uuid: str) -> None:
    """Reject a link: remember it and remove the edge from the graph."""
    db.add_rejected_edge(source_uuid, target_uuid)
