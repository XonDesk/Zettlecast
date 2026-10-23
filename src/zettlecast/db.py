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

    def upsert_note(self, note: NoteModel) -> None:
        """Insert or replace a note: chunk it, embed the chunks, store both."""
        from .embeddings import embed_documents

        chunks = create_chunks(note.full_text, note.uuid)
        # Prefix each chunk with the title so short chunks keep their context
        vectors = embed_documents([f"{note.title}\n\n{c.text}" for c in chunks])
        note_vector = vectors.mean(axis=0) if len(vectors) else np.zeros(DIM, np.float32)
        norm = np.linalg.norm(note_vector)
        if norm > 0:
            note_vector = note_vector / norm

        self.delete_note(note.uuid, keep_edges=True)

        self.notes.add([{
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
            "vector": note_vector.astype(np.float32).tolist(),
        }])
        if chunks:
            self.chunks.add([
                {
                    "chunk_id": c.chunk_id,
                    "note_uuid": note.uuid,
                    "text": c.text,
                    "start_char": c.start_char,
                    "vector": v.tolist(),
                }
                for c, v in zip(chunks, vectors)
            ])

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
        return [
            {
                "uuid": r["uuid"],
                "title": r["title"],
                "source_type": r["source_type"],
                "status": r["status"],
                "created_at": r["created_at"],
                "tags": json.loads(r["metadata_json"]).get("tags", []),
            }
            for r in rows[offset:offset + limit]
        ]

    def all_note_vectors(self) -> List[dict]:
        """Every note's graph inputs: uuid, title, type, source, created_at, metadata, text, vector."""
        rows = self.notes.search().select(
            ["uuid", "title", "source_type", "source_path", "created_at", "metadata_json", "full_text", "vector"]
        ).limit(100_000).to_list()
        return [
            {
                "uuid": r["uuid"],
                "title": r["title"],
                "source_type": r["source_type"],
                "source_path": r["source_path"],
                "created_at": r["created_at"],
                "metadata": json.loads(r["metadata_json"]),
                "full_text": r["full_text"],
                "vector": np.asarray(r["vector"], dtype=np.float32),
            }
            for r in rows
        ]

    def all_chunk_vectors(self) -> List[dict]:
        """note_uuid, text and vector of every chunk (passage matching)."""
        rows = self.chunks.search().select(["note_uuid", "text", "vector"]).limit(10_000_000).to_list()
        return [{"note_uuid": r["note_uuid"], "text": r["text"],
                 "vector": np.asarray(r["vector"], dtype=np.float32)} for r in rows]

    def delete_note(self, uuid: str, keep_edges: bool = False) -> None:
        self.notes.delete(f"uuid = {_q(uuid)}")
        self.chunks.delete(f"note_uuid = {_q(uuid)}")
        if not keep_edges:
            self.delete_edges_for_note(uuid)

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
