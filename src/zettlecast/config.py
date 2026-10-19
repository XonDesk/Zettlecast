"""
Zettlecast Configuration
Pydantic Settings for all configurable options.
"""

import secrets
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API Security ---
    api_token: str = Field(default_factory=lambda: secrets.token_urlsafe(32))

    # --- Storage Paths ---
    storage_path: Path = Field(default=Path.home() / "_BRAIN_STORAGE")
    lancedb_path: Path = Field(default=Path.home() / "_BRAIN_STORAGE" / ".lancedb")

    @property
    def podcasts_path(self) -> Path:
        """Downloaded/added podcast audio, one folder per show."""
        return self.storage_path / "podcasts"

    @property
    def transcripts_path(self) -> Path:
        """Transcript markdown notes written by the podcast pipeline."""
        return self.storage_path / "transcripts"

    # --- Embeddings & Search ---
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimensions: int = 1024
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    rerank_top_k: int = 50  # Chunk candidates to rerank
    return_top_k: int = 5  # Final results to return
    search_min_score: float = 0.05  # Drop reranked results below this relevance (0..1)
    model_idle_minutes: float = 5  # API server unloads the search models after this long unused (0 = keep)
    hf_token: str = Field(default="", validation_alias="HF_TOKEN")

    # --- Text Chunking (for embeddings) ---
    chunk_size: int = 1200  # Characters per chunk
    chunk_overlap: int = 150
    min_chunk_size: int = 100

    # --- Podcast Transcription (NeMo, CUDA) ---
    podcast_max_retries: int = 3
    nemo_chunk_duration_minutes: int = 60  # ASR chunk size; 60 min uses ~14 GB VRAM with local attention
    nemo_parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v2"  # English; use -v3 for 25 European languages
    nemo_diarization_model: str = "nvidia/Nemotron-3-Diarization"  # Sortformer, up to 8 speakers

    # --- LLM (Ollama) ---
    ollama_model: str = "qwen3.6:35b-a3b"  # Text + vision, 256K context
    ollama_base_url: str = "http://localhost:11434"
    llm_num_ctx: int = 32768  # Context window requested from Ollama
    llm_timeout_seconds: int = 600
    llm_seed: int | None = 42  # Fixed sampling seed: reprocessing gives the same note (None = random)
    llm_parallel: int = 4  # Transcript chunks in flight at once (overlaps generation with checking)

    # --- Features ---
    enable_auto_tagging: bool = True  # LLM tags + summary for non-podcast notes
    auto_tag_count: int = 5
    max_file_size_mb: int = 50
    use_marker_fallback: bool = True  # Marker for scanned/complex PDFs
    name_lookup: bool = True  # Look up unconfirmed names in podcast transcripts on Wikipedia (online)

    # --- Graph Construction ---
    graph_alpha: float = 0.7  # Weight for vector similarity
    graph_beta: float = 0.3  # Weight for tag overlap (Jaccard)
    graph_edge_threshold: float = 0.30  # Min composite score; unrelated docs score < 0.2 with Qwen3-Embedding
    graph_max_edges_per_note: int = 6

    # --- Server ---
    api_port: int = 8000
    api_host: str = "127.0.0.1"

    # --- Image Processing ---
    vision_model: str = "qwen3.6:35b-a3b"  # Same model as text: no model swapping
    max_image_size_mb: int = 25
    image_max_retries: int = 3

    def ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        for path in (self.storage_path, self.lancedb_path, self.podcasts_path, self.transcripts_path):
            path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
