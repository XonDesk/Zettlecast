"""
Graph Linker Module

Composite weighting algorithm for building graph edges between notes.
Combines vector similarity with tag overlap (Jaccard Index) and
optional temporal-based directed edges.
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from .config import settings
from .db import db
from .models import NoteModel

logger = logging.getLogger(__name__)


def calculate_jaccard_similarity(tags_a: List[str], tags_b: List[str]) -> float:
    """
    Calculate Jaccard similarity between two tag sets.
    
    Jaccard Index = |A ∩ B| / |A ∪ B|
    
    Args:
        tags_a: First tag list
        tags_b: Second tag list
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not tags_a and not tags_b:
        return 0.0
    
    set_a = set(t.lower().strip() for t in tags_a if t)
    set_b = set(t.lower().strip() for t in tags_b if t)
    
    if not set_a and not set_b:
        return 0.0
    
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec_a: First vector
        vec_b: Second vector
        
    Returns:
        Similarity score between -1.0 and 1.0
    """
    if not vec_a or not vec_b:
        return 0.0
    
    arr_a = np.array(vec_a, dtype=np.float32)
    arr_b = np.array(vec_b, dtype=np.float32)
    
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


def calculate_composite_weight(
    vector_sim: float,
    tag_sim: float,
    alpha: float = None,
    beta: float = None,
) -> float:
    """
    Calculate composite weight combining vector and tag similarity.
    
    Formula: α × vector_sim + β × tag_sim
    
    Args:
        vector_sim: Vector (cosine) similarity score
        tag_sim: Tag (Jaccard) similarity score
        alpha: Weight for vector similarity (default from settings)
        beta: Weight for tag similarity (default from settings)
        
    Returns:
        Composite score between 0.0 and 1.0
    """
    alpha = alpha if alpha is not None else settings.graph_alpha
    beta = beta if beta is not None else settings.graph_beta
    
    # Ensure weights sum to 1.0
    total = alpha + beta
    if total > 0:
        alpha = alpha / total
        beta = beta / total
    
    # Clamp vector_sim to [0, 1] since cosine can be negative
    vector_sim = max(0.0, min(1.0, vector_sim))
    
    composite = (alpha * vector_sim) + (beta * tag_sim)
    return round(composite, 4)


def get_temporal_direction(
    note_a: NoteModel,
    note_b: NoteModel,
) -> Tuple[str, str, bool]:
    """
    Determine edge direction based on creation dates.
    
    The older note points to the newer note (knowledge builds forward).
    
    Args:
        note_a: First note
        note_b: Second note
        
    Returns:
        Tuple of (source_uuid, target_uuid, is_directed)
    """
    if not settings.graph_temporal_direction:
        # Undirected - alphabetically order by UUID for consistency
        if note_a.uuid < note_b.uuid:
            return note_a.uuid, note_b.uuid, False
        return note_b.uuid, note_a.uuid, False
    
    # Directed: older → newer
    if note_a.created_at <= note_b.created_at:
        return note_a.uuid, note_b.uuid, True
    return note_b.uuid, note_a.uuid, True


class GraphEdge:
    """Represents an edge in the knowledge graph."""
    
    def __init__(
        self,
        source_uuid: str,
        target_uuid: str,
        weight: float,
        vector_sim: float,
        tag_sim: float,
        is_directed: bool = True,
        edge_type: str = "semantic",
    ):
        self.source_uuid = source_uuid
        self.target_uuid = target_uuid
        self.weight = weight
        self.vector_sim = vector_sim
        self.tag_sim = tag_sim
        self.is_directed = is_directed
        self.edge_type = edge_type
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        return {
            "source_uuid": self.source_uuid,
            "target_uuid": self.target_uuid,
            "weight": self.weight,
            "vector_sim": self.vector_sim,
            "tag_sim": self.tag_sim,
            "is_directed": self.is_directed,
            "edge_type": self.edge_type,
            "created_at": self.created_at.isoformat(),
        }


def build_edges_for_note(
    note_uuid: str,
    top_k: int = 10,
    threshold: float = None,
) -> List[GraphEdge]:
    """
    Build graph edges for a note using composite weighting.
    
    Pipeline:
    1. Vector search for top K candidates
    2. Calculate Jaccard similarity on tags
    3. Compute composite score
    4. Filter edges above threshold
    5. Apply temporal direction
    
    Args:
        note_uuid: UUID of the source note
        top_k: Number of candidates to consider
        threshold: Minimum composite score (default from settings)
        
    Returns:
        List of GraphEdge objects
    """
    threshold = threshold if threshold is not None else settings.graph_edge_threshold
    
    # Get source note
    source_note = db.get_note_by_uuid(note_uuid)
    if not source_note:
        logger.warning(f"Note not found: {note_uuid}")
        return []
    
    source_tags = source_note.metadata.tags or []
    
    # Vector search for candidates
    # Use first 1000 chars of text as query
    query_text = source_note.full_text[:1000]
    candidates = db.vector_search(query_text, top_k=top_k + 1)  # +1 because may include self
    
    # Filter out self
    candidates = [c for c in candidates if c.uuid != note_uuid]
    
    if not candidates:
        logger.debug(f"No candidates found for {note_uuid}")
        return []
    
    # Get rejected edges
    rejected = set(db.get_rejected_targets(note_uuid))
    
    edges = []
    for candidate in candidates[:top_k]:
        # Skip rejected edges
        if candidate.uuid in rejected:
            continue
        
        # Get full candidate note for tags and dates
        candidate_note = db.get_note_by_uuid(candidate.uuid)
        if not candidate_note:
            continue
        
        # Vector similarity is already computed by LanceDB
        # _distance is the L2 distance, convert to similarity
        # For cosine, we need to recompute or use a proxy
        # LanceDB uses L2 by default, so we'll use score as-is
        # (lower distance = higher similarity)
        vector_sim = 1.0 / (1.0 + candidate.score) if candidate.score >= 0 else 0.0
        
        # Calculate tag similarity
        candidate_tags = candidate_note.metadata.tags or []
        tag_sim = calculate_jaccard_similarity(source_tags, candidate_tags)
        
        # Calculate composite weight
        weight = calculate_composite_weight(vector_sim, tag_sim)
        
        # Check threshold
        if weight < threshold:
            logger.debug(f"Edge below threshold: {note_uuid} → {candidate.uuid} ({weight} < {threshold})")
            continue
        
        # Determine direction
        source_id, target_id, is_directed = get_temporal_direction(
            source_note, candidate_note
        )
        
        edge = GraphEdge(
            source_uuid=source_id,
            target_uuid=target_id,
            weight=weight,
            vector_sim=vector_sim,
            tag_sim=tag_sim,
            is_directed=is_directed,
            edge_type="semantic",
        )
        edges.append(edge)
        
        logger.debug(
            f"Edge created: {source_id} → {target_id} "
            f"(weight={weight}, vec={vector_sim:.3f}, tag={tag_sim:.3f})"
        )
    
    logger.info(f"Built {len(edges)} edges for note {note_uuid}")
    return edges


def check_llm_prerequisite(note_a: NoteModel, note_b: NoteModel) -> bool:
    """
    Use LLM to check if note_a is a prerequisite for understanding note_b.
    
    Args:
        note_a: Potential prerequisite note
        note_b: Note that may require note_a
        
    Returns:
        True if note_a is a prerequisite for note_b
    """
    if not settings.graph_llm_prerequisite:
        return False
    
    try:
        import httpx
        
        prompt = f"""Does understanding the concepts in the first text significantly help in understanding the second text?

Text 1 (potential prerequisite):
{note_a.full_text[:1500]}

Text 2:
{note_b.full_text[:1500]}

Answer with ONLY "YES" or "NO"."""

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            answer = response.json()["response"].strip().upper()
            
            return answer.startswith("YES")
    
    except Exception as e:
        logger.error(f"LLM prerequisite check failed: {e}")
        return False
