"""
Graph Linker: weighted edges between notes, and the notes' communities.

Notes are the only nodes. An edge's weight combines three signals:

- semantic: cosine of the note vectors after subtracting a *balanced* mean (the mean of
  each source group's mean: a podcast show, a website, the images). Embedding models make
  every text look somewhat alike (raw cosine 0.5-0.7 between any two episodes); centring
  removes that, and balancing keeps one big show from defining "average" for everything else
- passage: the share of each note's passages that closely match a passage of the other
  (passage vectors centred the same way). Passages repeated across a source group (intros,
  sponsor reads) are skipped. The best pair is kept as the link's reason
- entity: shared topics, people and organisations, weighted by rarity (IDF): the host who
  is in every episode adds nothing, a shared guest or a rare topic a lot

Each note keeps its k best partners (no global threshold: an episode pair and an
episode/screenshot pair live on different similarity scales); the edges are the union.
[[Wikilinks]] are always edges, even rejected; other rejected pairs never are. Communities: Leiden on the
weighted graph, ids kept stable across rebuilds, named after their most distinctive topics.
"""

import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import numpy as np

from .config import settings
from .db import db

logger = logging.getLogger(__name__)

_TIMESTAMP_RE = re.compile(r"^\[([\d.]+)s")
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
_ENTITY_KINDS = ("topic", "person", "org")


# --- Source groups and centring ---


def source_group(note: dict) -> str:
    """The source a note belongs to: a podcast show, a website, a subreddit, the images, else its type."""
    meta = note.get("metadata") or {}
    show = ((meta.get("custom") or {}).get("podcast") or {}).get("show")
    if show:
        return f"podcast:{show}"
    subreddit = ((meta.get("custom") or {}).get("reddit") or {}).get("subreddit")
    if subreddit:
        return f"reddit:r/{subreddit}"
    if note.get("source_type") == "web":
        host = urlparse(meta.get("source_url") or note.get("source_path") or "").netloc.lower()
        return f"web:{host.removeprefix('www.')}" if host else "web"
    return note.get("source_type") or "note"


def balanced_centre(vectors: np.ndarray, groups: List[str]) -> np.ndarray:
    """Subtract the mean of the per-group means (each group counts once), renormalise rows."""
    if len(vectors) == 0:
        return vectors
    by_group: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    mean = np.mean([vectors[idx].mean(axis=0) for idx in by_group.values()], axis=0)
    centred = vectors - mean
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    return centred / np.where(norms > 0, norms, 1)


# --- Entities ---


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().lower()


def topic_key(topic: str) -> str:
    """'Ultra running', 'ultrarunning', 'ultra-running' -> 'ultrarunning'; simple plurals folded."""
    key = re.sub(r"[^a-z0-9]", "", _fold(topic).replace("&", "and"))
    if len(key) > 5 and key.endswith("s") and not key.endswith("ss"):
        key = key[:-1]
    return key


def _people_keys(names: List[str]) -> Dict[str, str]:
    """
    Spelling -> canonical key: the same surname with a nearly identical first name and full
    name is one person ('Olaf Alexander Bu' / 'Olav Aleksander Bu', 'Ryan' / 'Ryne Anderson');
    'John' / 'Jane Smith' stay two.
    """
    import jellyfish

    def same(a: str, b: str) -> bool:
        return (jellyfish.jaro_winkler_similarity(a.split(" ")[0], b.split(" ")[0]) >= 0.85
                and jellyfish.jaro_winkler_similarity(a, b) >= 0.9)

    folded = {n: _fold(n) for n in names}
    keys: Dict[str, str] = {}
    by_surname: Dict[str, List[str]] = defaultdict(list)
    for name, f in folded.items():
        by_surname[f.split(" ")[-1]].append(name)
    for group in by_surname.values():
        group.sort(key=lambda n: folded[n])
        for name in group:
            match = next((keys[other] for other in group if other in keys and same(folded[name], folded[other])), None)
            keys[name] = match or folded[name]
    return keys


@dataclass
class EntityIndex:
    """Per note: entity key -> IDF weight; plus display names and the note x note similarity."""

    per_note: List[Dict[str, float]]
    display: Dict[str, str]
    similarity: np.ndarray


def note_entities(note: dict) -> Dict[str, List[str]]:
    """Raw topics / people / organisations of a note (people fall back to named speakers)."""
    meta = note.get("metadata") or {}
    custom = meta.get("custom") or {}
    people = custom.get("people") or list((custom.get("speaker_names") or {}).values())
    return {
        "topic": [str(t) for t in meta.get("tags") or []],
        "person": [str(p) for p in people if p and " " in str(p).strip()],
        "org": [str(o) for o in custom.get("organisations") or []],
    }


def build_entities(notes: List[dict]) -> EntityIndex:
    raw = [note_entities(n) for n in notes]
    person_keys = _people_keys(sorted({p for r in raw for p in r["person"]}))
    spellings: Dict[str, Counter] = defaultdict(Counter)
    keyed: List[Set[str]] = []
    for r in raw:
        keys = set()
        for kind in _ENTITY_KINDS:
            for value in r[kind]:
                base = person_keys[value] if kind == "person" else (
                    topic_key(value) if kind == "topic" else re.sub(r"[^a-z0-9]", "", _fold(value)))
                if not base:
                    continue
                key = f"{kind}:{base}"
                keys.add(key)
                spellings[key][value] += 1
        keyed.append(keys)

    n = len(notes)
    df = Counter(k for keys in keyed for k in keys)
    idf = {k: float(np.log((n + 1) / (c + 1))) for k, c in df.items()}
    per_note = [{k: idf[k] for k in keys} for keys in keyed]

    shared = sorted(k for k, c in df.items() if c >= 2 and idf[k] > 0)
    col = {k: i for i, k in enumerate(shared)}
    matrix = np.zeros((n, len(shared)), np.float32)
    norms = np.zeros(n, np.float32)
    for i, weights in enumerate(per_note):
        norms[i] = np.sqrt(sum(w * w for w in weights.values()))
        for k, w in weights.items():
            if k in col:
                matrix[i, col[k]] = w
    matrix /= np.where(norms > 0, norms, 1)[:, None]
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, 0)
    display = {k: c.most_common(1)[0][0] for k, c in spellings.items()}
    return EntityIndex(per_note, display, similarity)


# --- Passages ---


@dataclass
class Passages:
    """Chunk vectors grouped by note: rows offsets[i]:offsets[i+1] belong to note i."""

    vectors: np.ndarray
    texts: List[str]
    offsets: np.ndarray
    boilerplate: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    def rows(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))


def load_passages(notes: List[dict], chunks: List[dict]) -> Passages:
    by_note: Dict[str, List[dict]] = defaultdict(list)
    for c in chunks:
        by_note[c["note_uuid"]].append(c)
    vectors, texts, offsets = [], [], [0]
    for n in notes:
        for c in by_note.get(n["uuid"], []):
            vectors.append(c["vector"])
            texts.append(c["text"])
        offsets.append(len(texts))
    dim = settings.embedding_dimensions
    matrix = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, dim), np.float32)
    return Passages(matrix, texts, np.array(offsets), np.zeros(len(texts), bool))


def _best_per_note(sims: np.ndarray, passages: Passages, n_notes: int) -> np.ndarray:
    """(rows, chunks) similarities -> (rows, notes): each row's best match within each note."""
    best = np.full((sims.shape[0], n_notes), -1.0, np.float32)
    starts = passages.offsets[:-1]
    has = passages.offsets[1:] > starts
    if sims.shape[0] and has.any():
        best[:, has] = np.maximum.reduceat(sims, starts[has], axis=1)
    return best


def passage_shares(passages: Passages, groups: List[str], focus: Optional[int] = None) -> np.ndarray:
    """
    shares[i, j]: share of note i's passages that closely match some passage of note j.
    Also marks boilerplate: passages near-duplicated in many notes of their own group.
    focus: only row and column `focus` (one new note; no boilerplate marking).
    """
    n = len(groups)
    if focus is not None:
        return _focus_shares(passages, n, focus)
    best_rows: List[np.ndarray] = []
    for i in range(n):
        rows = passages.rows(i)
        sims = passages.vectors[rows] @ passages.vectors.T
        sims[:, rows] = -1.0  # own passages don't count
        best_rows.append(_best_per_note(sims, passages, n))

    members: Dict[str, np.ndarray] = {}
    for g in set(groups):
        members[g] = np.array([groups[j] == g for j in range(n)])
    shares = np.zeros((n, n), np.float32)
    for i, best in enumerate(best_rows):
        if not len(best):
            continue
        same = members[groups[i]].copy()
        same[i] = False
        if same.sum() >= 3:
            dup_notes = (best[:, same] >= settings.graph_boilerplate_similarity).sum(axis=1)
            boiler = dup_notes >= max(3, settings.graph_boilerplate_share * same.sum())
            passages.boilerplate[passages.rows(i)] = boiler
        else:
            boiler = np.zeros(len(best), bool)
        keep = best[~boiler]
        if len(keep):
            shares[i] = (keep >= settings.graph_passage_similarity).mean(axis=0)
    np.fill_diagonal(shares, 0)
    return shares


def _focus_shares(passages: Passages, n: int, i: int) -> np.ndarray:
    shares = np.zeros((n, n), np.float32)
    rows = passages.rows(i)
    if rows.start == rows.stop:
        return shares
    sims = passages.vectors[rows] @ passages.vectors.T
    sims[:, rows] = -1.0
    shares[i] = (_best_per_note(sims, passages, n) >= settings.graph_passage_similarity).mean(axis=0)
    matched = sims.max(axis=0) >= settings.graph_passage_similarity  # each passage vs note i's
    for j in range(n):
        rj = passages.rows(j)
        if j != i and rj.stop > rj.start:
            shares[j, i] = matched[rj].mean()
    shares[i, i] = 0
    return shares


def best_pair(passages: Passages, i: int, j: int) -> Optional[dict]:
    """The most alike pair of (non-boilerplate) passages of notes i and j."""
    ri, rj = passages.rows(i), passages.rows(j)
    if ri.start == ri.stop or rj.start == rj.stop:
        return None
    sims = passages.vectors[ri] @ passages.vectors[rj].T
    sims[passages.boilerplate[ri]] = -1
    sims[:, passages.boilerplate[rj]] = -1
    a, b = np.unravel_index(int(np.argmax(sims)), sims.shape)

    def side(index: int) -> dict:
        text = passages.texts[index]
        match = _TIMESTAMP_RE.match(text.lstrip())
        return {"text": text.strip()[:300], "time": float(match.group(1)) if match else None}

    return {"similarity": round(float(sims[a, b]), 3), "source": side(ri.start + a), "target": side(rj.start + b)}


# --- Wikilinks ---


def wikilink_pairs(notes: List[dict], referenced: bool = True) -> Set[Tuple[int, int]]:
    """(i, j) for every [[Title]] in note i that names note j, and (referenced) every note j a
    page i was fetched from."""
    by_title = {n["title"].strip().lower(): j for j, n in enumerate(notes)}
    by_uuid = {n.get("uuid"): j for j, n in enumerate(notes)}
    pairs = set()
    for i, n in enumerate(notes):
        referenced_by = (((n.get("metadata") or {}).get("custom") or {}).get("referenced_by") or []
                         if referenced else [])
        targets = [by_title.get(title.strip().lower()) for title in _WIKILINK_RE.findall(n.get("full_text") or "")]
        for j in targets + [by_uuid.get(uuid) for uuid in referenced_by]:
            if j is not None and j != i:
                pairs.add((min(i, j), max(i, j)))
    return pairs


# --- Edges ---


@dataclass
class Graph:
    notes: List[dict]
    weights: np.ndarray
    semantic: np.ndarray
    passage: np.ndarray
    shares: np.ndarray
    entities: EntityIndex
    passages: Passages
    links: Set[Tuple[int, int]]  # wikilinks and referenced_by: always edges, unless rejected
    typed: Set[Tuple[int, int]] = field(default_factory=set)  # [[wikilinks]]: edges even when rejected


def build(notes: Optional[List[dict]] = None, chunks: Optional[List[dict]] = None,
          focus: Optional[str] = None) -> Graph:
    """
    All pairwise signals for the corpus (notes and chunks from the database by default).
    focus: a note uuid; passage signals only for that note's row and column (incremental).
    """
    notes = db.all_note_vectors() if notes is None else notes
    chunks = db.all_chunk_vectors() if chunks is None else chunks
    groups = [source_group(n) for n in notes]
    dim = settings.embedding_dimensions
    vectors = np.stack([n["vector"] for n in notes]) if notes else np.zeros((0, dim), np.float32)
    centred = balanced_centre(vectors, groups)
    semantic = np.clip(centred @ centred.T, 0, 1)
    np.fill_diagonal(semantic, 0)
    passages = load_passages(notes, chunks)
    # passages are centred too: what all passages share ("interview talk": the host's usual
    # questions, greetings) would otherwise outscore two passages on the same subject
    owner = np.repeat(np.arange(len(notes)), np.diff(passages.offsets))
    passages.vectors = balanced_centre(passages.vectors, [groups[o] for o in owner])
    uuids = [n["uuid"] for n in notes]
    shares = passage_shares(passages, groups, uuids.index(focus) if focus in uuids else None)
    passage = (shares + shares.T) / 2
    entities = build_entities(notes)
    weights = (settings.graph_weight_semantic * semantic + settings.graph_weight_passage * passage
               + settings.graph_weight_entity * entities.similarity)
    return Graph(notes, weights, semantic, passage, shares, entities, passages,
                 wikilink_pairs(notes), wikilink_pairs(notes, referenced=False))


def _edge(graph: Graph, i: int, j: int, kind: str) -> dict:
    notes = graph.notes
    older, newer = sorted((i, j), key=lambda k: notes[k]["created_at"])
    shared = sorted(set(graph.entities.per_note[i]) & set(graph.entities.per_note[j]),
                    key=lambda k: -graph.entities.per_note[i][k])
    # name only what's distinctive: entities in under 30% of notes (the show, its host...)
    n = len(notes)
    min_idf = float(np.log((n + 1) / (0.3 * n + 1))) if n >= 10 else 0.0
    reason = {
        "shared": [{"kind": k.split(":", 1)[0], "name": graph.entities.display[k]}
                   for k in shared if graph.entities.per_note[i][k] > min_idf][:8],
    }
    if graph.passage[i, j] > 0:
        pair = best_pair(graph.passages, older, newer)
        if pair:
            reason["passage"] = pair
    weight = 1.0 if kind == "link" else float(graph.weights[i, j])
    return {
        "source_uuid": notes[older]["uuid"],
        "target_uuid": notes[newer]["uuid"],
        "weight": round(weight, 4),
        "kind": kind,
        "vector_sim": round(float(graph.semantic[i, j]), 4),
        "passage_sim": round(float(graph.passage[i, j]), 4),
        "entity_sim": round(float(graph.entities.similarity[i, j]), 4),
        "reason_json": json.dumps(reason, ensure_ascii=False),
    }


def _kind(graph: Graph, i: int, j: int) -> str:
    if (min(i, j), max(i, j)) in graph.links:
        return "link"
    if min(graph.shares[i, j], graph.shares[j, i]) >= settings.graph_duplicate_share:
        return "duplicate"
    return "similar"


def select_pairs(graph: Graph, rejected: Set[Tuple[str, str]], only: Optional[int] = None,
                 kth: Optional[Dict[str, float]] = None) -> Set[Tuple[int, int]]:
    """
    Union of each note's k best partners. only: just the pairs touching that note, which
    also joins another note's edges when it beats that note's current k-th best (kth: uuid ->
    weight of its k-th edge; missing = fewer than k edges).
    """
    k = settings.graph_max_edges_per_note
    uuids = [n["uuid"] for n in graph.notes]
    weights = graph.weights.copy()
    index = {u: i for i, u in enumerate(uuids)}
    rejected_pairs = set()
    for a, b in rejected:
        ia, ib = index.get(a), index.get(b)
        if ia is not None and ib is not None:
            weights[ia, ib] = weights[ib, ia] = -1
            rejected_pairs.add((min(ia, ib), max(ia, ib)))
    pairs = set()
    for i in range(len(uuids)) if only is None else [only]:
        order = np.argsort(-weights[i])
        for j in order[:k]:
            if j != i and weights[i, j] >= settings.graph_min_weight:
                pairs.add((min(i, int(j)), max(i, int(j))))
    if only is not None:
        # the new note may also be among another note's k best
        for j in range(len(uuids)):
            if j != only and weights[j, only] >= settings.graph_min_weight:
                if weights[j, only] >= (kth or {}).get(uuids[j], -np.inf):
                    pairs.add((min(j, only), max(j, only)))
    # a link the user typed ([[Title]]) always shows; one from referenced_by unless rejected
    return pairs | {p for p in graph.links
                    if (only is None or only in p) and (p in graph.typed or p not in rejected_pairs)}


def rebuild_graph() -> int:
    """Recompute every edge and the communities from scratch. Returns the edge count."""
    graph = build()
    pairs = select_pairs(graph, db.get_all_rejected_pairs())
    edges = [_edge(graph, i, j, _kind(graph, i, j)) for i, j in sorted(pairs)]
    db.clear_edges()
    db.add_edges(edges)
    communities = detect_communities(graph, edges, db.get_communities())
    db.replace_communities(communities)
    db.optimize()
    logger.info(f"Rebuilt graph: {len(graph.notes)} notes, {len(edges)} edges, {len(communities)} communities")
    return len(edges)


def update_edges_for_note(note_uuid: str) -> int:
    """Recompute the edges touching one note (after ingesting it); communities wait for a rebuild."""
    graph = build(focus=note_uuid)
    uuids = [n["uuid"] for n in graph.notes]
    if note_uuid not in uuids:
        return 0
    i = uuids.index(note_uuid)
    k = settings.graph_max_edges_per_note
    current: Dict[str, List[float]] = defaultdict(list)
    for e in db.get_all_edges():
        if note_uuid not in (e["source_uuid"], e["target_uuid"]) and e["kind"] != "link":
            current[e["source_uuid"]].append(e["weight"])
            current[e["target_uuid"]].append(e["weight"])
    kth = {u: sorted(w, reverse=True)[k - 1] for u, w in current.items() if len(w) >= k}
    pairs = select_pairs(graph, db.get_all_rejected_pairs(), only=i, kth=kth)
    edges = [_edge(graph, a, b, _kind(graph, a, b)) for a, b in sorted(pairs)]
    db.replace_edges_for_note(note_uuid, edges)
    logger.info(f"Linked '{graph.notes[i]['title']}' to {len(edges)} notes")
    return len(edges)


# --- Communities ---


def detect_communities(graph: Graph, edges: List[dict], previous: List[dict]) -> List[dict]:
    """
    Leiden on the weighted edges. Communities of fewer than 3 notes aren't kept (those notes
    show as unclustered). Ids follow the previous rebuild's communities by member overlap, so
    colours stay put; labels are the most distinctive topics.
    """
    import igraph as ig
    import leidenalg as la

    uuids = [n["uuid"] for n in graph.notes]
    if not edges:
        return []
    index = {u: i for i, u in enumerate(uuids)}
    g = ig.Graph(n=len(uuids), edges=[(index[e["source_uuid"]], index[e["target_uuid"]]) for e in edges])
    g.es["weight"] = [e["weight"] for e in edges]
    partition = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                                  resolution_parameter=settings.graph_resolution, seed=42)
    groups = [sorted(uuids[v] for v in members) for members in partition if len(members) >= 3]

    # stable ids: biggest overlaps with the previous communities first
    old = {int(c["community"]): set(c["members"]) for c in previous}
    candidates = sorted(
        ((len(set(m) & o), gi, cid) for gi, m in enumerate(groups) for cid, o in old.items() if set(m) & o),
        reverse=True,
    )
    assigned: Dict[int, int] = {}
    used: Set[int] = set()
    for overlap, gi, cid in candidates:
        if gi not in assigned and cid not in used and overlap / len(set(groups[gi]) | old[cid]) >= 0.3:
            assigned[gi] = cid
            used.add(cid)
    next_id = max([*old, *used, -1]) + 1
    out = []
    for gi, members in enumerate(groups):
        if gi not in assigned:
            assigned[gi] = next_id
            next_id += 1
        out.append({"community": assigned[gi], "label": community_label(graph, members), "members": members})
    return sorted(out, key=lambda c: -len(c["members"]))


def community_label(graph: Graph, members: List[str]) -> str:
    """The community's three most distinctive topics: share of its notes x IDF."""
    index = {n["uuid"]: i for i, n in enumerate(graph.notes)}
    counts: Counter = Counter()
    idf: Dict[str, float] = {}
    for u in members:
        for k, w in graph.entities.per_note[index[u]].items():
            if k.startswith("topic:"):
                counts[k] += 1
                idf[k] = w
    scored = sorted(counts, key=lambda k: -(counts[k] / len(members)) * idf[k])
    return " · ".join(graph.entities.display[k] for k in scored[:3])
