"""
Zettlecast Configuration
Pydantic Settings for all configurable options.
"""

import secrets
from pathlib import Path

from pydantic import Field, field_validator
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
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_num_ctx: int = 32768  # Context window requested from Ollama
    llm_timeout_seconds: int = 600
    llm_seed: int | None = 42  # Fixed sampling seed: reprocessing gives the same note (None = random)
    llm_parallel: int = 4  # Transcript chunks in flight at once (llama-server slots)
    # "llama-server": llama.cpp serves the Ollama model's own GGUF with parallel slots and
    # speculative decoding (~4x faster transcript cleanup); "ollama": Ollama only.
    # Falls back to Ollama when no llama-server.exe is found.
    llm_backend: str = "llama-server"
    llama_server_exe: str = ""  # default: newest %LOCALAPPDATA%\llama.cpp\*\llama-server.exe
    llama_server_port: int = 8081
    llama_server_ctx: int = 65536  # Context tokens shared by all slots (analysis needs up to ~32K)
    # n-gram drafts copy the transcript being cleaned; MTP drafts with the model's own extra layer
    llama_server_args: str = "--spec-type ngram-simple,draft-mtp"
    llm_idle_minutes: float = 5  # Stop llama-server after this long unused (frees ~27 GB VRAM)

    # --- Features ---
    enable_auto_tagging: bool = True  # LLM tags + summary for non-podcast notes
    auto_tag_count: int = 5
    max_file_size_mb: int = 50
    use_marker_fallback: bool = True  # Marker for scanned/complex PDFs
    name_lookup: bool = True  # Look up unconfirmed names in podcast transcripts on Wikipedia (online)

    # --- Graph Construction ---
    # weight = semantic * centred cosine + passage * matching passages + entity * shared entities
    graph_weight_semantic: float = 0.5
    graph_weight_passage: float = 0.3
    graph_weight_entity: float = 0.2
    graph_max_edges_per_note: int = 6  # Each note keeps its k best partners; edges = union of these
    graph_min_weight: float = 0.05  # Below this a "best partner" is no relation at all
    # passage thresholds are on centred chunk vectors (0.6 ~ the top 7% of each passage's best match)
    graph_passage_similarity: float = 0.6  # Chunk pairs at least this alike count as matching passages
    graph_boilerplate_similarity: float = 0.55  # Passages this alike repeated across a source group...
    graph_boilerplate_share: float = 0.2  # ...in this share of its notes (intros, sponsor reads) are skipped
    graph_duplicate_share: float = 0.9  # Notes sharing this share of passages are the same content
    graph_resolution: float = 1.0  # Leiden resolution: higher = more, smaller communities

    # --- Server ---
    api_port: int = 8000
    api_host: str = "127.0.0.1"
    web_lan: bool = False  # Web UI reachable from other devices on the network (sign-in; API stays local)

    # --- Image Processing ---
    vision_model: str = "qwen3.6:35b-a3b"  # Same model as text: no model swapping
    max_image_size_mb: int = 25
    image_max_retries: int = 3

    @field_validator("ollama_base_url")
    @classmethod
    def _ipv4_loopback(cls, url: str) -> str:
        # On Windows "localhost" tries IPv6 ::1 first; Ollama listens on IPv4 only, so every
        # request waited ~2 s for the fallback
        return url.replace("//localhost:", "//127.0.0.1:")

    def ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        for path in (self.storage_path, self.lancedb_path, self.podcasts_path, self.transcripts_path):
            path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
