"""
Zettlecast Configuration
Pydantic Settings for all configurable options.
"""

import secrets
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API Security ---
    api_token: str = Field(default_factory=lambda: secrets.token_urlsafe(32))

    # --- Storage Paths ---
    storage_path: Path = Field(default=Path.home() / "_BRAIN_STORAGE")
    lancedb_path: Path = Field(default=Path.home() / "_BRAIN_STORAGE" / ".lancedb")

    # --- Embedding Model ---
    # --- Embedding Model ---
    embedding_model: str = "google/embeddinggemma-300m"
    embedding_model_lite: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 768  # EmbeddingGemma-300M output dim (Correction from 1024)

    # --- Reranker ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 50  # Candidates to rerank
    return_top_k: int = 5  # Final results to return

    # --- Audio Transcription ---
    whisper_model: str = "large-v3-turbo"  # Fastest 2026 options: large-v3-turbo, distil-large-v3
    whisper_device: str = "auto"  # auto, cuda, cpu, mps
    hf_token: str = Field(default="", validation_alias="HF_TOKEN")  # Allow loading from env var
    podcast_max_retries: int = 3

    # --- ASR Backend Selection ---
    asr_backend: str = "auto"  # auto, nemo, parakeet-mlx, whisper
    diarization_backend: str = "auto"  # auto, pyannote, nemo, none

    # --- Mac (parakeet-mlx) ---
    parakeet_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    mlx_whisper_model: str = "mlx-community/whisper-large-v3-turbo"

    # --- NeMo Transcription (Container-based on Windows) ---
    use_nemo: bool = False  # Legacy flag, prefer asr_backend
    nemo_chunk_duration_minutes: int = 10  # Macro-chunk size for stable memory
    nemo_parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v2"  # Fast transcription
    nemo_diarization_model: str = "diar_msdd_telephonic"  # MSDD speaker diarization
    nemo_container_image: str = "zettlecast/nemo-asr:latest"
    nemo_container_auto_start: bool = True

    # --- LLM Provider ---
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    anthropic_model: str = "claude-3-haiku-20240307"
    anthropic_api_key: str = ""

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100

    # --- Features ---
    enable_context_enrichment: bool = False
    suggestion_cache_hours: int = 24
    max_file_size_mb: int = 50

    # --- Auto-Tagging (Enrichment) ---
    enable_auto_tagging: bool = True  # LLM tag extraction on ingest
    auto_tag_count: int = 5  # Number of tags to extract per note

    # --- Graph Construction ---
    graph_alpha: float = 0.7  # Weight for vector similarity
    graph_beta: float = 0.3  # Weight for tag overlap (Jaccard)
    graph_edge_threshold: float = 0.65  # Min score to create edge
    graph_temporal_direction: bool = True  # Older→newer directed edges
    graph_llm_prerequisite: bool = False  # LLM-based dependency check


    # --- Server ---
    api_port: int = 8000
    ui_port: int = 8501
    api_host: str = "0.0.0.0"

    # --- PDF Parsing ---
    pdf_parser_timeout: int = 60  # seconds
    use_marker_fallback: bool = True
    use_docling_fallback: bool = False  # Heavy, disabled by default

    # --- Image Processing ---
    vision_model: str = "qwen2.5-vl:7b"  # Qwen2.5-VL model (superior OCR vs LLaVA)
    max_image_size_mb: int = 25  # Max file size
    image_max_retries: int = 3  # Queue retry attempts

    def ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.lancedb_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
