"""
Graph Linker Module

Builds weighted edges between notes:

    weight = alpha * cosine(note vectors) + beta * jaccard(tags)

Edges point from the older note to the newer one. Each note keeps at most
graph_max_edges_per_note edges above graph_edge_threshold, and user-rejected
pairs are never linked.
"""

import logging
from typing import List

import numpy as np

from .config import settings
from .db import db

logger = logging.getLogger(__name__)


def calculate_jaccard_similarity(tags_a: List[str], tags_b: List[str]) -> float:
    """Jaccard index |A ∩ B| / |A ∪ B| of two tag lists (case-insensitive)."""
    set_a = {t.lower().strip() for t in tags_a if t}
    set_b = {t.lower().strip() for t in tags_b if t}
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def calculate_composite_weight(
    vector_sim: float,
    tag_sim: float,
    alpha: float = None,
    beta: float = None,
) -> float:
    """alpha * vector_sim + beta * tag_sim, with the weights normalised to sum to 1."""
    alpha = settings.graph_alpha if alpha is None else alpha
    beta = settings.graph_beta if beta is None else beta
    total = alpha + beta
    if total > 0:
        alpha, beta = alpha / total, beta / total
    vector_sim = max(0.0, min(1.0, vector_sim))
    return round(alpha * vector_sim + beta * tag_sim, 4)


def _edges_from(source: dict, others: List[dict], rejected: set) -> List[dict]:
    if not others:
        return []
    matrix = np.stack([o["vector"] for o in others])
    sims = matrix @ source["vector"]

    edges = []
    for other, vector_sim in zip(others, sims):
        if other["uuid"] == source["uuid"]:
            continue
        if tuple(sorted((source["uuid"], other["uuid"]))) in rejected:
            continue
        tag_sim = calculate_jaccard_similarity(source["tags"], other["tags"])
        weight = calculate_composite_weight(float(vector_sim), tag_sim)
        if weight < settings.graph_edge_threshold:
            continue
        older, newer = sorted((source, other), key=lambda n: n["created_at"])
        edges.append({
            "source_uuid": older["uuid"],
            "target_uuid": newer["uuid"],
            "weight": weight,
            "vector_sim": round(float(vector_sim), 4),
            "tag_sim": round(tag_sim, 4),
        })
    edges.sort(key=lambda e: e["weight"], reverse=True)
    return edges[: settings.graph_max_edges_per_note]


def update_edges_for_note(note_uuid: str) -> int:
    """Recompute the edges touching one note (call after ingesting it)."""
    notes = db.all_note_vectors()
    source = next((n for n in notes if n["uuid"] == note_uuid), None)
    if source is None:
        return 0
    edges = _edges_from(source, notes, db.get_all_rejected_pairs())
    db.replace_edges_for_note(note_uuid, edges)
    logger.info(f"Linked '{source['title']}' to {len(edges)} notes")
    return len(edges)


def rebuild_graph() -> int:
    """Recompute every edge from scratch. Returns the edge count."""
    notes = db.all_note_vectors()
    rejected = db.get_all_rejected_pairs()
    unique: dict = {}
    for source in notes:
        for edge in _edges_from(source, notes, rejected):
            key = tuple(sorted((edge["source_uuid"], edge["target_uuid"])))
            unique[key] = edge
    db.clear_edges()
    edges = list(unique.values())
    db.add_edges(edges)
    logger.info(f"Rebuilt graph: {len(notes)} notes, {len(edges)} edges")
    return len(edges)
