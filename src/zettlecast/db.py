"""
Zettlecast Database Layer (LanceDB)

Tables:
- notes:          one row per note, with a note-level vector (mean of its chunk
                  vectors) used for graph similarity
- chunks:         text chunks of every note with their own vectors; search runs
                  here so the whole of a long transcript is searchable
- graph_edges:    computed links between notes
- rejected_edges: user-rejected link suggestions
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import lancedb
import numpy as np
import pyarrow as pa

from .chunker import create_chunks
from .config import settings
from .models import NoteModel

logger = logging.getLogger(__name__)

DIM = settings.embedding_dimensions
# metadata.custom flag: stored on the primary without an index; the worker sends one (tagging.py)
INDEX_PENDING = "index_pending"
# the primary compacts the tables in the background after this many stored notes (the graph job's
# store_graph does it once per session anyway; a burst of writes in between kept 1.6 GB of versions)
COMPACT_AFTER_WRITES = 100

NOTES_SCHEMA = pa.schema([
    ("uuid", pa.string()),
    ("title", pa.string()),
    ("source_type", pa.string()),
    ("source_path", pa.string()),
    ("full_text", pa.string()),
    ("content_hash", pa.string()),
    ("status", pa.string()),
    ("created_at", pa.string()),
    ("updated_at", pa.string()),
    ("metadata_json", pa.string()),
    ("vector", pa.list_(pa.float32(), DIM)),
])

CHUNKS_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("note_uuid", pa.string()),
    ("text", pa.string()),
    ("start_char", pa.int32()),
    ("vector", pa.list_(pa.float32(), DIM)),
])

EDGES_SCHEMA = pa.schema([
    ("source_uuid", pa.string()),
    ("target_uuid", pa.string()),
    ("weight", pa.float32()),
    ("kind", pa.string()),  # similar | duplicate | link (a [[wikilink]])
    ("vector_sim", pa.float32()),  # centred cosine of the note vectors
    ("passage_sim", pa.float32()),  # share of passages that closely match the other note's
    ("entity_sim", pa.float32()),  # IDF-weighted overlap of topics, people, organisations
    ("reason_json", pa.string()),  # best passage pair, shared entities
    ("created_at", pa.string()),
])

COMMUNITIES_SCHEMA = pa.schema([
    ("community", pa.int32()),
    ("label", pa.string()),
    ("members", pa.list_(pa.string())),
])

REJECTED_SCHEMA = pa.schema([
    ("source_uuid", pa.string()),
    ("target_uuid", pa.string()),
    ("rejected_at", pa.string()),
])


def _q(value: str) -> str:
    """Quote a string literal for a LanceDB filter."""
    return "'" + value.replace("'", "''") + "'"


def _row_to_note(row: dict) -> NoteModel:
    return NoteModel(
        uuid=row["uuid"],
        title=row["title"],
        source_type=row["source_type"],
        source_path=row["source_path"],
        full_text=row["full_text"],
        content_hash=row["content_hash"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        metadata=json.loads(row["metadata_json"]),
    )



NOTE_VECTOR_COLUMNS = ["uuid", "title", "source_type", "source_path", "created_at", "metadata_json", "full_text",
                       "vector"]
CHUNK_VECTOR_COLUMNS = ["note_uuid", "text", "vector"]


def note_vector_rows(rows: List[dict]) -> List[dict]:
    """Note rows (NOTE_VECTOR_COLUMNS) as the graph takes them; a note still waiting for its first index is left out."""
    out = []
    for r in rows:
        vector = np.asarray(r["vector"], dtype=np.float32)
        if np.any(vector):
            out.append({"uuid": r["uuid"], "title": r["title"], "source_type": r["source_type"],
                        "source_path": r["source_path"], "created_at": r["created_at"],
                        "metadata": json.loads(r["metadata_json"]), "full_text": r["full_text"], "vector": vector})
    return out


def chunk_vector_rows(rows: List[dict]) -> List[dict]:
    return [{"note_uuid": r["note_uuid"], "text": r["text"], "vector": np.asarray(r["vector"], dtype=np.float32)}
            for r in rows]


def _note_row(note: NoteModel, vector) -> dict:
    return {
        "uuid": note.uuid,
        "title": note.title,
        "source_type": note.source_type,
        "source_path": note.source_path,
        "full_text": note.full_text,
        "content_hash": note.content_hash,
        "status": note.status,
        "created_at": note.created_at.isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "metadata_json": json.dumps(note.metadata.model_dump()),
        "vector": np.asarray(vector, np.float32).tolist(),
    }


def _chunks_and_vectors(note: NoteModel):
    """A note's chunks ({chunk_id, text, start_char}) and their embeddings, as upsert_note stores them."""
    from .embeddings import embed_documents

    chunks = create_chunks(note.full_text, note.uuid)
    # Prefix each chunk with the title so short chunks keep their context
    vectors = embed_documents([f"{note.title}\n\n{c.text}" for c in chunks])
    return [{"chunk_id": c.chunk_id, "text": c.text, "start_char": c.start_char} for c in chunks], vectors


def index_payload(note: NoteModel) -> dict:
    """
    What upsert_note needs to store a note without embedding it (the worker computes it on its GPU
    and sends it with the note): chunks, float32 vectors (base64) and a hash of the text they're from.
    """
    import base64

    from .identity import compute_content_hash

    chunks, vectors = _chunks_and_vectors(note)
    return {"chunks": chunks, "dim": DIM, "text_hash": compute_content_hash(note.full_text),
            "vectors": base64.b64encode(np.ascontiguousarray(vectors, np.float32).tobytes()).decode("ascii")}


def _from_payload(note: NoteModel, index: dict):
    """The chunks and vectors of an index_payload, or (None, None) when they don't fit this note."""
    import base64

    from .identity import compute_content_hash

    chunks = index.get("chunks") or []
    try:
        vectors = np.frombuffer(base64.b64decode(index.get("vectors") or ""), np.float32)
        vectors = vectors.reshape(len(chunks), DIM) if chunks else np.zeros((0, DIM), np.float32)
    except ValueError:
        vectors = None
    if (vectors is None or index.get("dim") != DIM or index.get("text_hash") != compute_content_hash(note.full_text)
            or any(not str(c.get("chunk_id", "")).startswith(note.uuid) for c in chunks)):
        logger.warning(f"Index sent for {note.title} doesn't match the note: embedding it here")
        return None, None
    return chunks, vectors


class Database:
    """Database wrapper for LanceDB operations."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.lancedb_path
        self._db = None
        self._tables: dict = {}

    def connect(self) -> "Database":
        self.db_path.mkdir(parents=True, exist_ok=True)
        # Every read checks for a newer table version: the API server must see what CLI runs
        # write (by default an open table stays at the version it was opened at)
        self._db = lancedb.connect(str(self.db_path), read_consistency_interval=timedelta(0))
        self._tables = {}
        return self

    @property
    def db(self):
        if self._db is None:
            self.connect()
        return self._db

    def _table(self, name: str, schema: pa.Schema):
        if name not in self._tables:
            if name in self.db.table_names():
                self._tables[name] = self.db.open_table(name)
            else:
                self._tables[name] = self.db.create_table(name, schema=schema)
        return self._tables[name]

    @property
    def notes(self):
        return self._table("notes", NOTES_SCHEMA)

    @property
    def chunks(self):
        return self._table("chunks", CHUNKS_SCHEMA)

    @property
    def graph_edges(self):
        if "graph_edges" not in self._tables and "graph_edges" in self.db.table_names():
            table = self.db.open_table("graph_edges")
            if table.schema.names != EDGES_SCHEMA.names:
                # edges are derived data: an older layout is dropped, the next rebuild refills it
                self.db.drop_table("graph_edges")
        return self._table("graph_edges", EDGES_SCHEMA)

    @property
    def communities(self):
        return self._table("communities", COMMUNITIES_SCHEMA)

    @property
    def rejected_edges(self):
        return self._table("rejected_edges", REJECTED_SCHEMA)

    # --- Notes ---

    def upsert_note(self, note: NoteModel, index: Optional[dict] = None) -> bool:
        """
        Insert or replace a note: chunk it, embed the chunks, store both. With index (from
        index_payload on the worker) the chunks and vectors are stored as sent, nothing embedded here.

        The primary never embeds: without a fitting index the note is stored waiting for one
        (_store_waiting). Returns whether the note was stored indexed (so it can be linked now).
        """
        chunks, vectors = _from_payload(note, index) if index else (None, None)
        if chunks is None and settings.is_primary:
            self._store_waiting(note)
            return False
        if chunks is None:
            chunks, vectors = _chunks_and_vectors(note)
        note.metadata.custom.pop(INDEX_PENDING, None)
        note_vector = vectors.mean(axis=0) if len(vectors) else np.zeros(DIM, np.float32)
        norm = np.linalg.norm(note_vector)
        if norm > 0:
            note_vector = note_vector / norm

        # one version per table (a delete and an add were two each): the note row replaced, its
        # chunks updated, added and the ones it no longer has deleted; its edges are kept
        self._replace_note_row(note, note_vector)
        rows = [{"chunk_id": c["chunk_id"], "note_uuid": note.uuid, "text": c["text"], "start_char": c["start_char"],
                 "vector": v.tolist()} for c, v in zip(chunks, vectors)]
        (self.chunks.merge_insert("chunk_id").when_matched_update_all().when_not_matched_insert_all()
         .when_not_matched_by_source_delete(f"note_uuid = {_q(note.uuid)}")
         .execute(rows or pa.Table.from_pylist([], schema=self.chunks.schema)))
        self._wrote()
        if settings.is_primary:  # the worker rebuilds the graph once its session's work is done
            from . import graphstate

            graphstate.mark_dirty()
        return True

    def store_placeholder(self, note: NoteModel) -> None:
        """A note row only (zero vector, no chunks): a page or file the worker hasn't read yet (ingestqueue)."""
        self.notes.add([_note_row(note, np.zeros(DIM, np.float32))])

    def _store_waiting(self, note: NoteModel) -> None:
        """
        The primary's store without an index: the note row only, marked for the worker. A note stored
        before keeps its chunks and vector (still searchable, with its old text) until the new index
        comes; a new one has none (listed, but not searchable or linked yet).
        """
        note.metadata.custom[INDEX_PENDING] = True
        old = self.notes.search().where(f"uuid = {_q(note.uuid)}").select(["vector"]).limit(1).to_list()
        vector = np.asarray(old[0]["vector"], np.float32) if old else np.zeros(DIM, np.float32)
        self._replace_note_row(note, vector)
        self._wrote()

    def _replace_note_row(self, note: NoteModel, vector) -> None:
        (self.notes.merge_insert("uuid").when_matched_update_all().when_not_matched_insert_all()
         .execute([_note_row(note, vector)]))

    def _wrote(self) -> None:
        """Count a stored note; on the primary, compact in the background every COMPACT_AFTER_WRITES."""
        if not settings.is_primary:
            return  # a standalone install's runs and rebuilds compact as before
        self._writes = getattr(self, "_writes", 0) + 1
        if self._writes < COMPACT_AFTER_WRITES or getattr(self, "_compacting", False):
            return
        self._writes, self._compacting = 0, True

        def compact() -> None:
            try:
                self.optimize()
            finally:
                self._compacting = False

        import threading

        threading.Thread(target=compact, name="db-compact", daemon=True).start()

    def update_note_text(self, uuid: str, full_text: str, content_hash: str) -> None:
        """A note's text without re-embedding it: for changes outside its indexed sections."""
        self.notes.update(where=f"uuid = {_q(uuid)}", values={
            "full_text": full_text, "content_hash": content_hash, "updated_at": datetime.utcnow().isoformat()})

    def update_note_metadata(self, uuid: str, metadata: dict) -> None:
        """A note's metadata without re-embedding it."""
        self.notes.update(where=f"uuid = {_q(uuid)}", values={
            "metadata_json": json.dumps(metadata), "updated_at": datetime.utcnow().isoformat()})

    def web_source_paths(self) -> List[tuple]:
        """(uuid, source_path) of every note stored from a web address."""
        rows = (self.notes.search().where("source_path LIKE 'http%'")
                .select(["uuid", "source_path"]).limit(100_000).to_list())
        return [(r["uuid"], r["source_path"]) for r in rows]

    def get_note_by_uuid(self, uuid: str) -> Optional[NoteModel]:
        rows = self.notes.search().where(f"uuid = {_q(uuid)}").limit(1).to_list()
        return _row_to_note(rows[0]) if rows else None

    def get_note_vector(self, uuid: str) -> Optional[np.ndarray]:
        rows = self.notes.search().where(f"uuid = {_q(uuid)}").select(["vector"]).limit(1).to_list()
        return np.asarray(rows[0]["vector"], dtype=np.float32) if rows else None

    def get_note_by_hash(self, content_hash: str) -> Optional[NoteModel]:
        rows = self.notes.search().where(f"content_hash = {_q(content_hash)}").limit(1).to_list()
        return _row_to_note(rows[0]) if rows else None

    def get_note_by_source_path(self, source_path: str) -> Optional[NoteModel]:
        rows = self.notes.search().where(f"source_path = {_q(source_path)}").limit(1).to_list()
        return _row_to_note(rows[0]) if rows else None

    def list_notes(
        self,
        status: Optional[str] = None,
        source_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[dict]:
        query = self.notes.search().select(
            ["uuid", "title", "source_type", "status", "created_at", "metadata_json"]
        )
        conditions = []
        if status:
            conditions.append(f"status = {_q(status)}")
        if source_type:
            conditions.append(f"source_type = {_q(source_type)}")
        if conditions:
            query = query.where(" AND ".join(conditions))

        rows = query.limit(100_000).to_list()
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        listed = []
        for r in rows[offset:offset + limit]:
            meta = json.loads(r["metadata_json"])
            custom = meta.get("custom") or {}
            listed.append({
                "uuid": r["uuid"],
                "title": r["title"],
                "source_type": r["source_type"],
                "status": r["status"],
                "created_at": r["created_at"],
                "tags": meta.get("tags", []),
                # for the worker's index, or a page / file it hasn't read yet (ingestqueue)
                "waiting": bool(custom.get(INDEX_PENDING) or custom.get("ingest_pending")),
                "add_failed": bool(custom.get("ingest_error")),
            })
        return listed

    def all_note_vectors(self) -> List[dict]:
        """
        Every indexed note's graph inputs: uuid, title, type, source, created_at, metadata, text, vector.
        A note still waiting for its first index (zero vector) isn't in the graph yet.
        """
        return note_vector_rows(self.notes.search().select(NOTE_VECTOR_COLUMNS).limit(100_000).to_list())

    def all_chunk_vectors(self) -> List[dict]:
        """note_uuid, text and vector of every chunk (passage matching)."""
        return chunk_vector_rows(self.chunks.search().select(CHUNK_VECTOR_COLUMNS).limit(10_000_000).to_list())

    def delete_note(self, uuid: str, keep_edges: bool = False) -> None:
        self.notes.delete(f"uuid = {_q(uuid)}")
        self.chunks.delete(f"note_uuid = {_q(uuid)}")
        if not keep_edges:
            self.delete_edges_for_note(uuid)
            if settings.is_primary:  # its communities change: the worker's next rebuild
                from . import graphstate

                graphstate.mark_dirty()

    # --- Chunk search ---

    def search_chunks(
        self,
        query_vector: np.ndarray,
        top_k: int = 50,
        exclude_note: Optional[str] = None,
    ) -> List[dict]:
        """Nearest chunks by cosine similarity. Returns rows with 'similarity'."""
        query = self.chunks.search(query_vector.tolist()).metric("cosine").limit(top_k)
        if exclude_note:
            query = query.where(f"note_uuid != {_q(exclude_note)}", prefilter=True)
        rows = query.select(["chunk_id", "note_uuid", "text", "start_char", "_distance"]).to_list()
        for r in rows:
            r["similarity"] = 1.0 - float(r.pop("_distance", 1.0))
        return rows

    def note_titles(self, uuids: List[str]) -> dict:
        """uuid -> (title, source_type) for the given notes."""
        if not uuids:
            return {}
        in_list = ", ".join(_q(u) for u in set(uuids))
        rows = self.notes.search().where(f"uuid IN ({in_list})").select(
            ["uuid", "title", "source_type"]
        ).limit(len(uuids)).to_list()
        return {r["uuid"]: (r["title"], r["source_type"]) for r in rows}

    # --- Graph edges ---

    def add_edges(self, edges: List[dict]) -> None:
        if edges:
            now = datetime.utcnow().isoformat()
            self.graph_edges.add([{**e, "created_at": now} for e in edges])

    def replace_edges_for_note(self, uuid: str, edges: List[dict]) -> None:
        self.delete_edges_for_note(uuid)
        self.add_edges(edges)

    def get_all_edges(self) -> List[dict]:
        return self.graph_edges.search().limit(1_000_000).to_list()

    def get_edges_for_note(self, uuid: str) -> List[dict]:
        return self.graph_edges.search().where(
            f"source_uuid = {_q(uuid)} OR target_uuid = {_q(uuid)}"
        ).limit(1000).to_list()

    def delete_edges_for_note(self, uuid: str) -> None:
        self.graph_edges.delete(f"source_uuid = {_q(uuid)} OR target_uuid = {_q(uuid)}")

    def clear_edges(self) -> None:
        self.graph_edges.delete("true")

    # --- Communities ---

    def replace_communities(self, communities: List[dict]) -> None:
        self.communities.delete("true")
        if communities:
            self.communities.add(communities)

    def get_communities(self) -> List[dict]:
        return self.communities.search().limit(100_000).to_list()

    # --- Rejected edges ---

    def add_rejected_edge(self, source_uuid: str, target_uuid: str) -> None:
        a, b = sorted((source_uuid, target_uuid))
        self.rejected_edges.add([{
            "source_uuid": a,
            "target_uuid": b,
            "rejected_at": datetime.utcnow().isoformat(),
        }])
        self.graph_edges.delete(
            f"(source_uuid = {_q(a)} AND target_uuid = {_q(b)}) OR "
            f"(source_uuid = {_q(b)} AND target_uuid = {_q(a)})"
        )

    def get_rejected_targets(self, uuid: str) -> set:
        rows = self.rejected_edges.search().where(
            f"source_uuid = {_q(uuid)} OR target_uuid = {_q(uuid)}"
        ).limit(100_000).to_list()
        return {r["target_uuid"] if r["source_uuid"] == uuid else r["source_uuid"] for r in rows}

    def get_all_rejected_pairs(self) -> set:
        rows = self.rejected_edges.search().limit(1_000_000).to_list()
        return {(r["source_uuid"], r["target_uuid"]) for r in rows}

    # --- Maintenance ---

    def optimize(self, keep: timedelta = timedelta(minutes=10)) -> None:
        """
        Merge small data files and drop table versions older than `keep`. Every write adds a
        version and its own files, so without this the tables grow far past their data (1.1 GB
        for ~200 MB). Recent versions stay: another process (API server, CLI run) may be reading one.
        """
        for name in self.db.table_names():
            try:
                self.db.open_table(name).optimize(cleanup_older_than=keep)
            except Exception as e:  # a concurrent write can win; the next rebuild tries again
                logger.warning(f"Optimizing table {name} failed: {e}")

    # --- Stats ---

    def stats(self) -> dict:
        notes = self.notes.search().select(["source_type"]).limit(1_000_000).to_list()
        by_type: dict = {}
        for n in notes:
            by_type[n["source_type"]] = by_type.get(n["source_type"], 0) + 1
        return {
            "notes": len(notes),
            "chunks": self.chunks.count_rows(),
            "edges": self.graph_edges.count_rows(),
            "rejected_edges": self.rejected_edges.count_rows(),
            "by_source_type": by_type,
        }


# Global database instance
db = Database()
