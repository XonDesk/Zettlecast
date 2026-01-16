# Zettlecast

**Digital Zettelkasten Middleware** - A local-first AI knowledge system with semantic search, automatic link suggestions, and immutable identity.

## Requirements

- **Python 3.11 or 3.12** (required for NeMo/ML library compatibility)
  - ⚠️ Python 3.13+ is NOT supported
- **FFmpeg** (for audio transcription)
- **CUDA GPU** (recommended for NeMo transcription, optional)

## Quick Start

### macOS / Linux

```bash
# Install Python 3.11 if needed
# macOS: brew install python@3.11
# Linux: sudo apt install python3.11 python3.11-venv

# Clone the repo
git clone https://github.com/XonDesk/Zettlecast.git
cd Zettlecast

# Run setup (installs Python deps + Ollama)
chmod +x setup.sh
./setup.sh

# Start the server
./run.sh
```

### Windows

```powershell
# Requires Python 3.11 - download from:
# https://www.python.org/downloads/release/python-3119/

# Clone the repo
git clone https://github.com/XonDesk/Zettlecast.git
cd Zettlecast

# Run setup in PowerShell
.\setup.ps1

# Start the server
.\.venv\Scripts\Activate.ps1
zettlecast serve
```

### All Platforms

Then open:
- **UI**: http://localhost:8501
- **API**: http://localhost:8000/docs

## Features

- 📄 **PDF Ingestion** - Tiered parsing (pypdf → Marker → Docling)
- 🌐 **Web Clipping** - Ingest any URL via bookmarklet
- 🎙️ **Audio Transcription** - Faster-Whisper (default) or NVIDIA NeMo (parallel)
- 🔍 **Semantic Search** - EmbeddingGemma-300M + BGE reranker
- 🔗 **Link Suggestions** - AI-powered "Gardener" finds related notes
- 📊 **Graph View** - Visualize connections with Cytoscape.js

## CLI Commands

```bash
# Start server
zettlecast serve

# Ingest files
zettlecast ingest /path/to/files

# Quick-add URL
zettlecast add https://example.com/article

# Search
zettlecast search "machine learning"

# Get your bookmarklet
zettlecast token

# View stats
zettlecast stats

# Podcast transcription
zettlecast podcast add /path/to/audio/files
zettlecast podcast run
zettlecast podcast status
```

## Configuration

Copy `.env.example` to `.env` and customize.

### Podcast Transcription

Zettlecast supports two transcription pipelines for audio processing:

#### 1. Whisper Pipeline (Default)
- **Model**: `faster-whisper` (large-v3-turbo by default)
- **Pros**: Lightweight, works on CPU, simple setup
- **Cons**: No speaker diarization, slower on large files
- **Setup**: Included by default, no extra installation needed

**Usage:**
```bash
zettlecast podcast add /path/to/audio
zettlecast podcast run  # Uses Whisper by default
```

#### 2. NeMo Pipeline (Faster & Parallel)
- **Models**: 
  - `nvidia/parakeet-tdt-0.6b-v2` for transcription
  - `diar_msdd_telephonic` for speaker diarization
- **Pros**: Fast (60 min audio in ~2 sec), speaker diarization, word-level timestamps, parallel processing
- **Cons**: Requires more disk space (~5GB), Python 3.11/3.12 only, GPU recommended
- **Pipeline**: Audio chunked into 10-min segments, transcription & diarization run in parallel per chunk

**Requirements for NeMo:**
- **Python 3.11 or 3.12** (NeMo does not support Python 3.13+)
- **CUDA GPU** (recommended, CPU works but slower)
- **~5GB disk space** for models
- **FFmpeg** (required for audio processing)

**Installation:**

Option 1: During setup (recommended)
```bash
# The setup script will ask if you want to install podcast support
./setup.sh      # macOS/Linux
.\setup.ps1     # Windows
```

Option 2: Manual installation
```bash
# Activate your virtual environment first
source .venv/bin/activate  # macOS/Linux
# or
.\.venv\Scripts\Activate.ps1  # Windows

# Install podcast dependencies
pip install -e ".[podcast]"

# Windows only: Apply NeMo compatibility patch
python scripts/patch_nemo_windows.py
```

**macOS/Linux Note:**
The setup script automatically applies compatibility checks for NeMo on your platform.

**Windows Note:**
NeMo has a known issue with `signal.SIGKILL` on Windows. The setup script automatically applies a patch. If you install manually, run `python scripts/patch_nemo_windows.py` after installing.

**Enable NeMo:**
Add to your `.env` file:
```env
USE_NEMO=true
```

Or use the command-line flag:
```bash
# Use NeMo for this run even if USE_NEMO=false
zettlecast podcast run --use-nemo
```

### Standard Configuration

```env
# Storage
STORAGE_PATH=~/_BRAIN_STORAGE
LANCEDB_PATH=~/_BRAIN_STORAGE/.lancedb

# Embedding & Search
EMBEDDING_MODEL=google/embeddinggemma-300m
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Audio Transcription
# Default pipeline: Whisper (faster-whisper, included)
WHISPER_MODEL=large-v3-turbo
WHISPER_DEVICE=cuda  # or 'cpu' for CPU-only

# NeMo Pipeline (Optional - Faster & Parallel + Speaker Diarization)
# Requires: pip install -e ".[podcast]"
USE_NEMO=false  # Set to true to use NeMo instead of Whisper
NEMO_CHUNK_DURATION_MINUTES=10
NEMO_PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v2
NEMO_DIARIZATION_MODEL=diar_msdd_telephonic

# LLM Provider (for context enrichment & podcast enhancement)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest?url=...` | Ingest URL |
| POST | `/ingest` (file) | Upload file |
| GET | `/notes` | List all notes |
| GET | `/notes/{uuid}` | Get note with suggestions |
| GET | `/search?q=...` | Semantic search |
| POST | `/notes/{uuid}/link` | Accept/reject link |

All endpoints require `?token=YOUR_API_TOKEN`.

## Bookmarklet

Run `zettlecast token` to get your bookmarklet, then:
1. Create a new bookmark
2. Set the URL to the JavaScript code
3. Click it on any page to save to Zettlecast

## Architecture

```
Zettlecast/
├── src/zettlecast/
│   ├── main.py      # FastAPI endpoints
│   ├── config.py    # Settings
│   ├── db.py        # LanceDB operations
│   ├── parser.py    # PDF/Web/Audio parsing
│   ├── chunker.py   # Text splitting
│   ├── search.py    # Vector search + reranking
│   ├── identity.py  # UUID management
│   └── cli.py       # CLI commands
└── ~/_BRAIN_STORAGE/
    ├── *.md         # Your notes
    └── .lancedb/    # Vector database
```

## License

MIT
