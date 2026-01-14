# Zettlecast

**Digital Zettelkasten Middleware** - A local-first AI knowledge system with semantic search, automatic link suggestions, and immutable identity.

## Quick Start

### macOS / Linux

```bash
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
- 🎙️ **Audio Transcription** - Faster-Whisper with timestamps
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
```

## Configuration

Copy `.env.example` to `.env` and customize:

```env
# Storage
STORAGE_PATH=~/_BRAIN_STORAGE

# Models
EMBEDDING_MODEL=google/embeddinggemma-300m
WHISPER_MODEL=medium

# LLM (for context enrichment)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
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
