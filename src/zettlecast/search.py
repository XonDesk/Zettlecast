"""
Zettlecast Search & Retrieval
Vector search with reranking and rejection filtering.
"""

import logging
from typing import List, Optional

from sentence_transformers import CrossEncoder

from .config import settings
from .db import db
from .models import LinkSuggestion, SearchResult

logger = logging.getLogger(__name__)

# Lazy-loaded reranker
_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """Get or load the reranker model (cached)."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker: {settings.reranker_model}")
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank_results(
    query: str,
    results: List[SearchResult],
    top_k: int = None,
) -> List[SearchResult]:
    """
    Rerank search results using cross-encoder.
    
    Args:
        query: The search query
        results: Initial results from vector search
        top_k: Number of results to return after reranking
        
    Returns:
        Reranked and filtered results
    """
    top_k = top_k or settings.return_top_k
    
    if not results:
        return []
    
    reranker = get_reranker()
    
    # Prepare pairs for reranking
    pairs = [(query, r.snippet) for r in results]
    
    # Get reranker scores
    scores = reranker.predict(pairs)
    
    # Combine with results and sort
    scored_results = list(zip(results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    # Update scores and return top_k
    final_results = []
    for result, score in scored_results[:top_k]:
        result.score = float(score)
        final_results.append(result)
    
    return final_results


def search(
    query: str,
    top_k: int = None,
    rerank: bool = True,
    status_filter: Optional[str] = None,
) -> List[SearchResult]:
    """
    Full search pipeline: vector search → rerank → return.
    
    Args:
        query: Search query text
        top_k: Number of final results to return
        rerank: Whether to apply reranking
        status_filter: Optional status filter (inbox, reviewed, archived)
        
    Returns:
        List of SearchResult
    """
    top_k = top_k or settings.return_top_k
    
    # Step 1: Vector search (get more candidates than needed)
    candidates = db.vector_search(
        query_text=query,
        top_k=settings.rerank_top_k if rerank else top_k,
        status_filter=status_filter,
    )
    
    if not candidates:
        return []
    
    # Step 2: Rerank (optional)
    if rerank and len(candidates) > top_k:
        results = rerank_results(query, candidates, top_k=top_k)
    else:
        results = candidates[:top_k]
    
    return results


def get_suggestions_for_note(
    note_uuid: str,
    use_cache: bool = True,
) -> List[LinkSuggestion]:
    """
    Get link suggestions for a note.
    
    Args:
        note_uuid: UUID of the note to get suggestions for
        use_cache: Whether to use cached suggestions
        
    Returns:
        List of LinkSuggestion
    """
    # Check cache first
    if use_cache:
        cached = db.get_cached_suggestions(note_uuid)
        if cached:
            suggestions = []
            for uuid, score in zip(cached.suggested_uuids, cached.scores):
                note = db.get_note_by_uuid(uuid)
                if note:
                    suggestions.append(LinkSuggestion(
                        uuid=uuid,
                        title=note.title,
                        score=score,
                    ))
            return suggestions
    
    # Get the source note
    source_note = db.get_note_by_uuid(note_uuid)
    if not source_note:
        return []
    
    # Search for similar notes
    candidates = db.vector_search(
        query_text=source_note.full_text[:1000],  # Use first 1000 chars as query
        top_k=settings.rerank_top_k,
    )
    
    # Remove self
    candidates = [c for c in candidates if c.uuid != note_uuid]
    
    if not candidates:
        return []
    
    # Rerank
    reranked = rerank_results(
        query=source_note.title + " " + source_note.full_text[:500],
        results=candidates,
        top_k=10,  # Get more than needed for filtering
    )
    
    # Filter rejected edges
    rejected = set(db.get_rejected_targets(note_uuid))
    filtered = [r for r in reranked if r.uuid not in rejected]
    
    # Take top 3
    final = filtered[:3]
    
    # Build suggestions
    suggestions = [
        LinkSuggestion(
            uuid=r.uuid,
            title=r.title,
            score=r.score,
        )
        for r in final
    ]
    
    # Cache results
    if suggestions:
        db.cache_suggestions(
            note_uuid=note_uuid,
            suggested_uuids=[s.uuid for s in suggestions],
            scores=[s.score for s in suggestions],
        )
    
    return suggestions


def accept_link(source_uuid: str, target_uuid: str) -> bool:
    """
    Accept a suggested link by adding wikilink to source file.
    
    Returns:
        True if link was added, False if already exists
    """
    from pathlib import Path
    from .identity import add_wikilink_to_file
    
    source_note = db.get_note_by_uuid(source_uuid)
    target_note = db.get_note_by_uuid(target_uuid)
    
    if not source_note or not target_note:
        return False
    
    # Add wikilink to source file
    source_path = Path(source_note.source_path)
    if source_path.exists():
        return add_wikilink_to_file(source_path, target_note.title)
    
    return False


def reject_link(source_uuid: str, target_uuid: str) -> None:
    """
    Reject a suggested link (adds to rejected_edges).
    """
    db.add_rejected_edge(source_uuid, target_uuid)
    
    # Invalidate cache
    db.suggestion_cache.delete(f"note_uuid = '{source_uuid}'")
