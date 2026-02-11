# Zettlecast Agent Reference

Complete reference for AI agents operating the Zettlecast system.

## Project Overview

**Zettlecast** is a privacy-first Digital Zettelkasten with podcast transcription, image analysis, and semantic search.

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.12 + FastAPI |
| Frontend | Next.js 16 + React 19 + TypeScript |
| Database | LanceDB (768-dim embeddings) |
| Embedding | `google/embeddinggemma-300m` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| ASR | NeMo Parakeet-TDT + MSDD diarization, faster-whisper fallback |
| LLM | Ollama (default: `llama3.2:3b`) |
| Vision | Ollama (`qwen2.5-vl:7b`) |

### Directory Structure

```
src/zettlecast/           # Backend source
  main.py                 # FastAPI app, all API endpoints
  config.py               # Settings (from .env)
  db.py                   # LanceDB operations
  models.py               # Pydantic data models (NoteModel, ChunkModel, etc.)
  parser.py               # File/URL ingestion pipeline
  search.py               # Semantic search + reranking
  chunker.py              # Text chunking for embeddings
  identity.py             # UUID/frontmatter management
  cli.py                  # Click CLI commands
  podcast/                # Podcast transcription pipeline
    nemo_transcriber.py   # NeMo Parakeet + MSDD transcription
    aligner.py            # Word-to-speaker alignment
    enhancer.py           # LLM cleanup, summary, key points
    formatter.py          # Markdown output formatting
    models.py             # Podcast-specific Pydantic models
    queue.py              # Transcription job queue
    transcriber_factory.py # Backend selection (nemo/whisper/auto)
  image/                  # Image processing pipeline
    image_parser.py       # Vision model analysis
    queue.py              # Image job queue
frontend/                 # Next.js frontend
  app/                    # App router pages
    notes/                # Note list + detail views
    podcasts/             # Podcast queue management
    images/               # Image queue management
    search/               # Semantic search UI
    graph/                # Knowledge graph visualization
    settings/             # Configuration UI
  components/             # Shared components
    Sidebar.tsx           # Navigation sidebar
    Graph.tsx             # Force-directed graph
    MarkdownContent.tsx   # Markdown renderer
  lib/
    api.ts                # API client
    types.ts              # TypeScript type definitions
    utils.ts              # Formatting utilities
tests/                    # pytest test suite
```

---

## Setup & Running

### Initial Setup

```bash
# Clone and install
git clone https://github.com/XonDesk/Zettlecast.git
cd Zettlecast
./setup.sh          # Creates .venv, installs deps, generates .env
```

### Start Backend

```bash
# Option 1: CLI server
source .venv/bin/activate
zettlecast serve --port 8000

# Option 2: Shortcut wrapper
./zc serve
```

### Start Frontend

```bash
cd frontend
npm install
npm run dev          # Starts on http://localhost:3000
```

### Environment

- Backend API: `http://localhost:8000`
- Frontend: `http://localhost:3000`
- API docs: `http://localhost:8000/docs`
- API token: auto-generated in `.env` (see `API_TOKEN`)

---

## CLI Commands

All commands can be run via `zettlecast <command>` or `./zc <command>`.

### Core Commands

| Command | Description |
|---------|-------------|
| `serve [--port PORT]` | Start API server (default port 8000) |
| `ingest PATH` | Ingest a file or directory (PDF, markdown, audio) |
| `search QUERY [--top-k K]` | Semantic search (default top_k=5) |
| `add URL` | Quick-add a URL for ingestion |
| `token` | Print API token and bookmarklet JS |
| `stats` | Show database statistics |

### Podcast Commands (`zettlecast podcast ...`)

| Command | Flags | Description |
|---------|-------|-------------|
| `add PATH` | `--name NAME`, `--recursive/--no-recursive` | Add audio files to transcription queue |
| `import URL` | `--limit N` (default 5) | Import episodes from RSS feed |
| `run` | `--limit N`, `--no-enhance`, `--backend BACKEND`, `--no-sync` | Process pending episodes |
| `status` | | Show queue status and time estimate |
| `retry` | | Retry failed episodes |

**Backend options for `podcast run --backend`:**
- `nemo` - NeMo Parakeet-TDT + MSDD diarization (GPU recommended)
- `whisper` - faster-whisper (CPU-friendly)
- `parakeet-mlx` - Parakeet on Apple Silicon
- `mlx-whisper` - Whisper on Apple Silicon
- (omit) - auto-detect best available

**`podcast run` pipeline:**
1. Sync queue with storage (unless `--no-sync`)
2. Select transcription backend
3. For each pending episode:
   - Transcribe audio (word-level timestamps + diarization)
   - Align words to speakers, merge micro-segments, merge minor speakers
   - LLM enhancement (unless `--no-enhance`): cleanup, keywords, summary, key points, chapters
   - Save as markdown with YAML frontmatter
   - Auto-ingest into LanceDB

### Image Commands (`zettlecast image ...`)

| Command | Flags | Description |
|---------|-------|-------------|
| `add PATH` | `--collection NAME`, `--recursive/--no-recursive` | Add images to processing queue |
| `run` | `--limit N`, `--model MODEL` | Process images with vision model |
| `status` | | Show queue status |
| `retry` | | Retry failed images |

---

## API Endpoints

All endpoints require `?token=API_TOKEN` query parameter (except `/health`, `/docs`).

### Health

```
GET /health
Response: { "status": "healthy", "version": "0.1.0" }
```

### Notes

```
POST /ingest?url=URL
POST /ingest (multipart file upload)
Response: { "status": "success"|"duplicate"|"failed", "uuid": str, "title": str }

GET /notes?status=inbox&source_type=audio&limit=50&offset=0
Response: { "notes": [...], "count": int }

GET /notes/{uuid}?include_suggestions=true
Response: {
    "uuid": str,
    "title": str,
    "source_type": "pdf"|"web"|"audio"|"markdown"|"rss"|"image",
    "source_path": str,
    "full_text": str,       // Markdown content with YAML frontmatter
    "status": "inbox"|"reviewed"|"archived",
    "created_at": str,
    "metadata": {
        "author": str,
        "tags": [str],
        "source_url": str,
        "language": str,
        "word_count": int,
        "page_count": int,
        "duration_seconds": int,
        "embedded_media": [str],
        "custom": {}
    },
    "suggestions": [{ "uuid": str, "title": str, "score": float, "reason": str }]
}

DELETE /notes/{uuid}
Response: { "status": "deleted", "uuid": str }
```

### Search

```
GET /search?q=QUERY&top_k=5&rerank=true
Response: {
    "query": str,
    "results": [{ "uuid": str, "title": str, "score": float, "snippet": str, "source_type": str }],
    "count": int
}
```

### Links

```
POST /notes/{uuid}/link
Body: { "target_uuid": str, "action": "accept"|"reject" }
Response: { "status": "linked"|"already_linked"|"rejected" }
```

### Graph

```
GET /graph?limit=2000
Response: {
    "nodes": [{ "id": str, "name": str, "source_type": str, "val": int }],
    "links": [{ "source": str, "target": str, "value": float }]
}
```

### Podcast Processing

```
GET /podcast/status
Response: {
    "by_status": { "pending": int, "processing": int, "completed": int, "review": int },
    "total": int,
    "estimated_remaining": str,
    "items": [{ "job_id": str, "podcast_name": str, "episode_title": str, "status": str, "added_at": str, "error_message": str|null, "attempts": int }]
}

POST /podcast/import
Body: { "feed_url": str, "limit": int }
Response: { "status": "success", "added_count": int, "job_ids": [str], "message": str }

POST /podcast/run
Body: { "limit": int, "backend": str|null }
Response: { "status": "started"|"already_running"|"no_pending", "pending_count": int, "limit": int, "message": str }

GET /podcast/running
Response: { "is_running": bool, "current_episode": str|null, "processed_count": int, "error_count": int, "started_at": str|null }

POST /podcast/retry
Response: { "status": "success", "retried_count": int, "message": str }

POST /podcast/sync
Response: { "status": "success", "sync_stats": {...}, "message": str }

POST /podcast/reset-stuck
Response: { "status": "success", "reset_count": int, "message": str }
```

### Image Processing

```
GET /image/status
Response: { "by_status": {...}, "total": int, "estimated_remaining": str, "items": [...] }

POST /image/add
Body: { "path": str, "collection_name": str|null }
Response: { "status": "success"|"duplicate", "added_count": int, "job_ids": [str], "message": str }

POST /image/run
Body: { "limit": int, "model": str|null }
Response: { "status": "started"|"already_running"|"no_pending", ... }

GET /image/running
Response: { "is_running": bool, "current_image": str|null, "processed_count": int, "error_count": int, "started_at": str|null }

POST /image/retry
Response: { "status": "success", "retried_count": int, "message": str }
```

### Settings

```
GET /settings
Response: {
    "embedding_model": str,
    "reranker_model": str,
    "whisper_model": str,
    "llm_provider": "ollama"|"openai"|"anthropic",
    "ollama_model": str,
    "enable_context_enrichment": bool,
    "chunk_size": int,
    "storage_path": str,
    "asr_backend": str,
    "hf_token": "***"|""
}

POST /settings
Body: { key: value, ... }  // Any subset of settings above
Response: { "status": "updated", "updated_keys": [str], "message": str }
```

---

## Configuration (.env)

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `API_TOKEN` | auto-generated | API authentication token |
| `STORAGE_PATH` | `~/_BRAIN_STORAGE` | Root storage for notes and data |
| `LANCEDB_PATH` | `~/_BRAIN_STORAGE/.lancedb` | Vector database path |

### Embedding & Search

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `google/embeddinggemma-300m` | Embedding model (768-dim) |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `RERANK_TOP_K` | `50` | Candidates to rerank |
| `RETURN_TOP_K` | `5` | Final results returned |

### Audio Transcription

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_BACKEND` | `auto` | `auto`, `nemo`, `parakeet-mlx`, `whisper` |
| `WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model |
| `WHISPER_DEVICE` | `auto` | `auto`, `cuda`, `cpu`, `mps` |
| `DIARIZATION_BACKEND` | `auto` | `auto`, `pyannote`, `nemo`, `none` |
| `HF_TOKEN` | (empty) | HuggingFace token (for pyannote diarization) |
| `PODCAST_MAX_RETRIES` | `3` | Max retries per episode |

### NeMo

| Variable | Default | Description |
|----------|---------|-------------|
| `NEMO_CHUNK_DURATION_MINUTES` | `10` | Audio chunk size |
| `NEMO_PARAKEET_MODEL` | `nvidia/parakeet-tdt-0.6b-v2` | Transcription model |
| `NEMO_DIARIZATION_MODEL` | `diar_msdd_telephonic` | Speaker diarization model |

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `anthropic` |
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama model for enhancement |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model (if provider=openai) |
| `ANTHROPIC_MODEL` | `claude-3-haiku-20240307` | Anthropic model (if provider=anthropic) |

### Chunking

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `MIN_CHUNK_SIZE` | `100` | Minimum chunk size |

### Features

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_CONTEXT_ENRICHMENT` | `false` | LLM-powered context prefix for chunks |
| `ENABLE_AUTO_TAGGING` | `false` | LLM tag extraction |
| `AUTO_TAG_COUNT` | `5` | Tags per note |

### Image Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_MODEL` | `qwen2.5-vl:7b` | Ollama vision model |
| `MAX_IMAGE_SIZE_MB` | `25` | Max image file size |

### Graph

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPH_ALPHA` | `0.7` | Vector similarity weight |
| `GRAPH_BETA` | `0.3` | Tag overlap weight |
| `GRAPH_EDGE_THRESHOLD` | `0.65` | Min score for edge |

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `8000` | API server port |
| `API_HOST` | `0.0.0.0` | API bind address |

---

## Processing Pipelines

### Podcast Transcription Pipeline

```
Audio File
  → Chunk into 10-min segments (NeMo) or full file (Whisper)
  → Transcribe (word-level timestamps)
  → Diarize (speaker labels via MSDD or pyannote)
  → Align words to speakers (aligner.py)
  → Merge micro-segments (<1.5s bounces)
  → Merge minor speakers (auto-detect over-segmentation)
  → LLM Enhancement (enhancer.py):
      - Cleanup: fix ASR errors, domain terms, filler words
      - Extract keywords (5-10)
      - Detect chapters/sections
      - Generate summary (3-5 sentences)
      - Extract key points (5-10 concrete insights)
      - Flag uncertain corrections with [[text??]] markers
  → Format as markdown with YAML frontmatter (formatter.py)
  → Auto-ingest into LanceDB
```

### Note Ingestion Pipeline

```
Source (URL / File / Markdown)
  → Parse content (trafilatura for web, pypdf for PDF, etc.)
  → Generate UUID + frontmatter
  → Create text chunks (512 chars, 50 overlap)
  → Compute embeddings (768-dim)
  → Store in LanceDB (notes table + chunks with vectors)
```

### Search Pipeline

```
Query
  → Embed query text
  → Vector similarity search (top 50 candidates)
  → Cross-encoder reranking (top 5 results)
  → Return with scores and snippets
```

---

## Data Model

### Note (Markdown with YAML Frontmatter)

```yaml
---
uuid: "550e8400-e29b-41d4-a716-446655440000"
title: "Episode Title"
source_type: audio          # pdf | web | audio | markdown | rss | image
source: /path/to/file
status: inbox               # inbox | reviewed | archived
created: "2026-01-15T10:30:00"
duration_seconds: 3600      # audio only
language: en
speakers: 2                 # audio only
tags:
- keyword1
- keyword2
podcast:                    # audio only
  show: "Show Name"
  episode: "Episode Title"
---

## Summary
3-5 sentence summary of content.

## Key Points
- Concrete insight 1
- Concrete insight 2

## Chapters
- **Introduction** ([0s]): Description
- **Main Topic** ([120s]): Description

## Transcript
[0.0s - 45.2s] speaker_0: Transcribed text here...
[45.3s - 90.1s] speaker_1: More text...
```

### LanceDB Tables

- **notes** - Full note records with embedding vectors
- **graph_edges** - Accepted link relationships between notes
- **rejected_edges** - User-rejected link suggestions (negative constraints)

---

## Frontend Architecture

### Pages

| Route | File | Purpose |
|-------|------|---------|
| `/` | redirect | Redirects to `/notes` |
| `/notes` | `app/notes/page.tsx` | Note list with status + source_type filters |
| `/notes/[uuid]` | `app/notes/[uuid]/page.tsx` | Note detail with markdown rendering |
| `/podcasts` | `app/podcasts/page.tsx` | Podcast queue management |
| `/images` | `app/images/page.tsx` | Image queue management |
| `/search` | `app/search/page.tsx` | Semantic search interface |
| `/graph` | `app/graph/page.tsx` | Knowledge graph visualization |
| `/settings` | `app/settings/page.tsx` | Configuration UI |

### Key Libraries

- `react-markdown` + `remark-gfm` - Markdown rendering
- `react-force-graph-2d` - Graph visualization
- `clsx` - Conditional CSS classes

### API Client

All API calls go through `frontend/lib/api.ts` which appends the auth token automatically. The API URL is configured via `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`) and token via `NEXT_PUBLIC_API_TOKEN` in `frontend/.env.local`.

### Styling

Custom CSS in `frontend/app/globals.css` using CSS variables for a dark theme. No Tailwind utility classes in components - all styles are class-based (`.card`, `.btn`, `.badge`, `.markdown-content`, etc.).

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_aligner.py -v      # Speaker alignment + merging
pytest tests/test_enhancer.py -v     # LLM enhancement (mocked)
pytest tests/test_chunker.py -v      # Text chunking
pytest tests/test_identity.py -v     # UUID/frontmatter
pytest tests/test_parser.py -v       # File parsing

# Frontend build check
cd frontend && npm run build
```

---

## Common Operations

### Transcribe a podcast from RSS

```bash
./zc podcast import https://example.com/feed.xml --limit 3
./zc podcast run
```

### Ingest a batch of PDFs

```bash
./zc ingest /path/to/papers/
```

### Search the knowledge base

```bash
./zc search "gut training endurance"
```

### Check system health

```bash
curl http://localhost:8000/health
./zc stats
./zc podcast status
```
