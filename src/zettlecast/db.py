"""
Zettlecast Database Layer
LanceDB operations for notes and embeddings.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

from .config import settings
from .models import NoteModel, RejectedEdge, SearchResult, SuggestionCache


# Initialize embedding function from registry
def get_embedding_function():
    """Get the sentence-transformers embedding function."""
    import os
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
        
    model = get_registry().get("sentence-transformers")
    return model.create(name=settings.embedding_model, device="cpu")


# Initialize embedding function
func = get_embedding_function()

# LanceDB schema with embedded vector field
class NoteTable(LanceModel):
    """LanceDB table schema for notes with auto-embedding."""
    
    uuid: str
    title: str
    source_type: str
    source_path: str
    full_text: str = func.SourceField()  # Mark as source for embedding
    content_hash: str
    status: str
    created_at: str  # ISO format string
    updated_at: str
    metadata_json: str  # JSON serialized metadata
    chunks_json: str  # JSON serialized chunks
    
    # Vector field - auto-populated from full_text
    vector: Vector(settings.embedding_dimensions) = func.VectorField(default=None)


class RejectedEdgeTable(LanceModel):
    """LanceDB table schema for rejected edges."""
    
    source_uuid: str
    target_uuid: str
    rejected_at: str


class SuggestionCacheTable(LanceModel):
    """LanceDB table schema for suggestion cache."""
    
    note_uuid: str
    suggested_uuids_json: str  # JSON array
    scores_json: str  # JSON array
    cached_at: str


class GraphEdgeTable(LanceModel):
    """LanceDB table schema for computed graph edges."""
    
    source_uuid: str
    target_uuid: str
    weight: float  # Composite score
    vector_sim: float  # Vector similarity component
    tag_sim: float  # Jaccard tag similarity component
    is_directed: bool  # True if temporal direction applied
    edge_type: str  # "semantic" | "prerequisite"
    created_at: str


class Database:
    """Database wrapper for LanceDB operations."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.lancedb_path
        self._db = None
        self._notes_table = None
        self._rejected_table = None
        self._cache_table = None
    
    def connect(self) -> "Database":
        """Connect to LanceDB."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.db_path))
        return self
    
    @property
    def db(self):
        if self._db is None:
            self.connect()
        return self._db
    
    def _get_or_create_table(self, name: str, schema):
        """Get existing table or create new one."""
        if name in self.db.table_names():
            return self.db.open_table(name)
        return self.db.create_table(name, schema=schema)
    
    @property
    def notes(self):
        """Get or create notes table."""
        if self._notes_table is None:
            self._notes_table = self._get_or_create_table("notes", NoteTable)
        return self._notes_table
    
    @property
    def rejected_edges(self):
        """Get or create rejected edges table."""
        if self._rejected_table is None:
            self._rejected_table = self._get_or_create_table("rejected_edges", RejectedEdgeTable)
        return self._rejected_table
    
    @property
    def suggestion_cache(self):
        """Get or create suggestion cache table."""
        if self._cache_table is None:
            self._cache_table = self._get_or_create_table("suggestion_cache", SuggestionCacheTable)
        return self._cache_table
    
    # --- Note Operations ---
    
    def upsert_note(self, note: NoteModel) -> None:
        """Insert or update a note."""
        import json
        
        # Convert to table format
        record = NoteTable(
            uuid=note.uuid,
            title=note.title,
            source_type=note.source_type,
            source_path=note.source_path,
            full_text=note.full_text,
            content_hash=note.content_hash,
            status=note.status,
            created_at=note.created_at.isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            metadata_json=json.dumps(note.metadata.model_dump()),
            chunks_json=json.dumps([c.model_dump() for c in note.chunks]),
        )
        
        # Check if exists
        existing = self.get_note_by_uuid(note.uuid)
        if existing:
            # Update (convert to dict for update)
            self.notes.update(
                where=f"uuid = '{note.uuid}'",
                values=record.model_dump(exclude={"vector"}), # Vector update might be tricky if not recalculated? 
            )
        else:
            # Insert
            self.notes.add([record])
    
    def get_note_by_uuid(self, uuid: str) -> Optional[NoteModel]:
        """Get a note by UUID."""
        import json
        
        results = self.notes.search().where(f"uuid = '{uuid}'").limit(1).to_list()
        if not results:
            return None
        
        row = results[0]
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
            chunks=[json.loads(c) for c in json.loads(row["chunks_json"])],
        )
    
    def get_note_by_hash(self, content_hash: str) -> Optional[NoteModel]:
        """Get a note by content hash (for deduplication)."""
        results = self.notes.search().where(f"content_hash = '{content_hash}'").limit(1).to_list()
        if not results:
            return None
        return self.get_note_by_uuid(results[0]["uuid"])
    
    def get_note_by_source_path(self, source_path: str) -> Optional[NoteModel]:
        """Get a note by source path/URL (for URL deduplication)."""
        # Escape single quotes in URL
        escaped_path = source_path.replace("'", "''")
        results = self.notes.search().where(f"source_path = '{escaped_path}'").limit(1).to_list()
        if not results:
            return None
        return self.get_note_by_uuid(results[0]["uuid"])
    
    def list_notes(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[dict]:
        """List notes with optional status filter."""
        query = self.notes.search()
        
        if status:
            query = query.where(f"status = '{status}'")
        
        results = query.limit(limit + offset).to_list()
        
        return [
            {
                "uuid": r["uuid"],
                "title": r["title"],
                "source_type": r["source_type"],
                "status": r["status"],
                "created_at": r["created_at"],
            }
            for r in results[offset:offset + limit]
        ]
    
    def delete_note(self, uuid: str) -> bool:
        """Delete a note by UUID."""
        self.notes.delete(f"uuid = '{uuid}'")
        return True
    
    # --- Search Operations ---
    
    def vector_search(
        self,
        query_text: str,
        top_k: int = 50,
        status_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform vector similarity search."""
        search = self.notes.search(query_text).limit(top_k)
        
        if status_filter:
            search = search.where(f"status = '{status_filter}'")
        
        results = search.to_list()
        
        return [
            SearchResult(
                uuid=r["uuid"],
                title=r["title"],
                score=float(r.get("_distance", 0)),
                snippet=r["full_text"][:200] + "...",
                source_type=r["source_type"],
            )
            for r in results
        ]
    
    # --- Rejected Edges ---
    
    def add_rejected_edge(self, source_uuid: str, target_uuid: str) -> None:
        """Add a rejected edge (user said "don't suggest this link")."""
        # Ensure consistent ordering
        if source_uuid > target_uuid:
            source_uuid, target_uuid = target_uuid, source_uuid
        
        record = {
            "source_uuid": source_uuid,
            "target_uuid": target_uuid,
            "rejected_at": datetime.utcnow().isoformat(),
        }
        self.rejected_edges.add([record])
    
    def is_edge_rejected(self, source_uuid: str, target_uuid: str) -> bool:
        """Check if an edge has been rejected."""
        # Check both directions
        if source_uuid > target_uuid:
            source_uuid, target_uuid = target_uuid, source_uuid
        
        results = self.rejected_edges.search().where(
            f"source_uuid = '{source_uuid}' AND target_uuid = '{target_uuid}'"
        ).limit(1).to_list()
        
        return len(results) > 0
    
    def get_rejected_targets(self, source_uuid: str) -> List[str]:
        """Get all UUIDs that are rejected for a given source."""
        results = self.rejected_edges.search().where(
            f"source_uuid = '{source_uuid}' OR target_uuid = '{source_uuid}'"
        ).to_list()
        
        rejected = set()
        for r in results:
            if r["source_uuid"] == source_uuid:
                rejected.add(r["target_uuid"])
            else:
                rejected.add(r["source_uuid"])
        
        return list(rejected)
    
    # --- Suggestion Cache ---
    
    def get_cached_suggestions(self, note_uuid: str) -> Optional[SuggestionCache]:
        """Get cached suggestions if still valid."""
        import json
        
        results = self.suggestion_cache.search().where(
            f"note_uuid = '{note_uuid}'"
        ).limit(1).to_list()
        
        if not results:
            return None
        
        row = results[0]
        cached_at = datetime.fromisoformat(row["cached_at"])
        
        # Check if cache is still valid
        if datetime.utcnow() - cached_at > timedelta(hours=settings.suggestion_cache_hours):
            return None
        
        return SuggestionCache(
            note_uuid=row["note_uuid"],
            suggested_uuids=json.loads(row["suggested_uuids_json"]),
            scores=json.loads(row["scores_json"]),
            cached_at=cached_at,
        )
    
    def cache_suggestions(
        self,
        note_uuid: str,
        suggested_uuids: List[str],
        scores: List[float],
    ) -> None:
        """Cache suggestions for a note."""
        import json
        
        # Delete old cache
        self.suggestion_cache.delete(f"note_uuid = '{note_uuid}'")
        
        # Add new cache
        record = {
            "note_uuid": note_uuid,
            "suggested_uuids_json": json.dumps(suggested_uuids),
            "scores_json": json.dumps(scores),
            "cached_at": datetime.utcnow().isoformat(),
        }
        self.suggestion_cache.add([record])


# Global database instance
db = Database()
