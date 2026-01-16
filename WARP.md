# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

Zettlecast is a local-first AI knowledge system with semantic search, automatic link suggestions, and immutable identity. It's a digital Zettelkasten middleware built with FastAPI, LanceDB, and Streamlit.

## Development Environment

**Python Requirements**: Python 3.10+
**Virtual Environment**: `.venv` (activated automatically by setup scripts)

### Setup

**Windows**:
```powershell
.\setup.ps1
```

**macOS/Linux**:
```bash
chmod +x setup.sh
./setup.sh
```

## Common Commands

### Running the Application

**Windows**:
```powershell
.\.venv\Scripts\Activate.ps1
zettlecast serve
```

**macOS/Linux**:
```bash
./run.sh
```

Servers:
- API: http://localhost:8000
- UI: http://localhost:8501

### CLI Commands

```bash
# Ingest files
zettlecast ingest /path/to/files

# Quick-add URL
zettlecast add https://example.com/article

# Search notes
zettlecast search "query" -k 5

# Get API token and bookmarklet
zettlecast token

# View database statistics
zettlecast stats

# Start server (API + UI)
zettlecast serve

# Start API only
zettlecast serve --no-ui
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_chunker.py

# Run with coverage
pytest --cov=src/zettlecast --cov-report=term-missing
```

### Code Quality

```bash
# Lint code
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Type checking
mypy src/zettlecast
```

## Architecture

### Core Modules

**`main.py`**: FastAPI application with endpoints for ingestion, search, and note management. Includes token-based authentication middleware.

**`db.py`**: LanceDB wrapper layer. Manages three tables:
- `notes`: Main note storage with auto-embedded vectors
- `rejected_edges`: User-rejected link suggestions (anti-recommendations)
- `suggestion_cache`: Cached link suggestions with TTL

**`parser.py`**: Multi-source content extraction:
- **PDF**: Tiered parsing (pypdf → Marker → Docling fallback)
- **Web**: Trafilatura for main content extraction
- **Audio**: Faster-Whisper transcription

**`search.py`**: Two-stage retrieval pipeline:
1. Vector search with EmbeddingGemma-300M (returns top 50 candidates)
2. CrossEncoder reranking with BGE-reranker-v2-m3 (returns top K)
3. Rejection filtering (removes previously rejected links)

**`chunker.py`**: Recursive text splitter with semantic boundaries (paragraphs → sentences → words). Default: 512 chars with 50 char overlap.

**`identity.py`**: UUID management and frontmatter operations. Ensures all notes have immutable UUIDs in YAML frontmatter. Handles wikilink insertion.

**`models.py`**: Pydantic models for notes, chunks, search results, and link suggestions.

**`config.py`**: Settings loaded from `.env` file using pydantic-settings.

### Vector Database Schema

**Notes Table** (LanceDB):
- `uuid` (str): Primary identifier
- `title` (str): Note title
- `source_type` (str): pdf | web | audio | markdown | rss
- `source_path` (str): Original file path or URL
- `full_text` (str): Complete extracted text (used for embedding)
- `vector` (Vector[1024]): Auto-generated embedding from full_text
- `content_hash` (str): SHA256 for deduplication
- `status` (str): inbox | reviewed | archived
- `metadata_json` (str): Serialized NoteMetadata
- `chunks_json` (str): Serialized list of ChunkModel

### Content Processing Pipeline

1. **Ingestion** (`/ingest` endpoint or CLI):
   - Parse content using appropriate parser (PDF/Web/Audio)
   - Generate UUID and compute content hash
   - Check for duplicates via content_hash
   - Create chunks using recursive splitter
   - Store in LanceDB (embeddings auto-generated)
   - Create markdown file in STORAGE_PATH

2. **Link Suggestions** ("Gardener" system):
   - Use note's full_text (first 1000 chars) as query
   - Vector search for similar notes (top 50)
   - Rerank with cross-encoder
   - Filter out self and rejected edges
   - Cache results for SUGGESTION_CACHE_HOURS

3. **Search**:
   - User query → embedding → vector search
   - Rerank candidates with query/snippet pairs
   - Return top K results with scores

### Podcast Module

Located in `src/zettlecast/podcast/`, this is an optional feature for advanced audio transcription:

**Key Files**:
- `nemo_transcriber.py`: Parallel pipeline with NVIDIA NeMo (transcription + diarization)
- `chunker.py`: Audio macro-chunking for stable memory usage
- `aligner.py`: Word-to-speaker alignment logic
- `transcriber.py`: Legacy Faster-Whisper integration
- `vad.py`: Voice Activity Detection preprocessing
- `enhancer.py`: LLM-based transcript enhancement (summaries, keywords)
- `queue.py`: Batch processing queue for multiple episodes
- `formatter.py`: Transcript output formatting
- `models.py`: Podcast-specific data models

**Installation**: `pip install -e '.[podcast]'` (NeMo requires additional setup)

## Configuration

**`.env` file** (copy from `.env.example`):

Key settings:
- `API_TOKEN`: Auto-generated on first run
- `STORAGE_PATH`: Where markdown notes are saved (default: `~/_BRAIN_STORAGE`)
- `LANCEDB_PATH`: Vector database location (default: `~/_BRAIN_STORAGE/.lancedb`)
- `EMBEDDING_MODEL`: Default is `google/embeddinggemma-300m`
- `RERANKER_MODEL`: Default is `BAAI/bge-reranker-v2-m3`
- `LLM_PROVIDER`: ollama | openai | anthropic (for context enrichment)
- `ENABLE_CONTEXT_ENRICHMENT`: false by default (adds LLM summaries to chunks, slow)

## Important Patterns

### UUID Management
- Every note MUST have a UUID in its frontmatter
- UUIDs are immutable (content-independent)
- Use `identity.py` functions for frontmatter operations
- Wikilinks use note titles, not UUIDs

### Deduplication
- Content hash (SHA256 of full_text) prevents duplicate ingestion
- Check `db.get_note_by_hash()` before creating new notes

### Tiered PDF Parsing
- Start with fast pypdf extraction
- If chars_per_page < 500 and `USE_MARKER_FALLBACK=true`, use Marker
- Docling is available but disabled by default (heavy)

### Rejection Tracking
- When users reject link suggestions, store in `rejected_edges` table
- Always filter rejected edges when showing suggestions
- Rejections invalidate suggestion cache for that note

### LanceDB Embedding
- Embeddings are auto-generated by LanceDB's sentence-transformers function
- The `full_text` field is used as the source for embedding
- No need to manually compute or store vectors

### Error Handling in Parsers
- All parsers return `ProcessingResult` with status: success | partial | failed
- Include `error_type` and `error_message` for debugging
- Failed ingestions should not crash the server

## File Structure

```
src/zettlecast/
├── main.py          # FastAPI app and endpoints
├── config.py        # Settings from .env
├── db.py            # LanceDB operations
├── parser.py        # PDF/Web/Audio parsing
├── chunker.py       # Text splitting
├── search.py        # Vector search + reranking
├── identity.py      # UUID and frontmatter
├── models.py        # Pydantic models
├── cli.py           # CLI commands
├── ui/              # Streamlit UI
│   └── app.py
└── podcast/         # Optional: advanced transcription
    ├── transcriber.py
    ├── vad.py
    ├── enhancer.py
    ├── queue.py
    ├── formatter.py
    └── models.py

tests/
├── conftest.py
├── test_chunker.py
└── test_identity.py
```

## Testing Notes

- Test fixtures in `conftest.py` add src to path
- Tests use temporary directories for file operations
- Mock LanceDB connections when testing parsers in isolation
- Identity tests verify frontmatter parsing and UUID generation
- Chunker tests verify recursive splitting and overlap behavior
