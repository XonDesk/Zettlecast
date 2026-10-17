# Zettlecast

A private, local-first Zettelkasten for Windows with an NVIDIA GPU. It turns
podcasts into speaker-labelled, searchable transcripts and links everything you
save (podcasts, PDFs, web pages, markdown notes and images) into a knowledge graph.
Nothing leaves your machine except name lookups on Wikipedia while checking
transcripts (`NAME_LOOKUP=false` turns them off).

## What it does

```
Podcast audio ─► Parakeet-TDT (speech→text, word timestamps)
              ─► Nemotron-3-Diarization (who spoke when, up to 8 speakers)
              ─► Qwen3.6 via Ollama (summary, key points, chapters, speaker names, cleanup)
              ─► every name fix / cleanup edit checked against the audio (Parakeet as judge)
              ─► Markdown note in _BRAIN_STORAGE/transcripts/
              ─► Chunked + embedded (Qwen3-Embedding) into LanceDB
              ─► Linked to related notes in the graph

PDF / web page / markdown / image ─► text (pypdf, Marker, Trafilatura, Qwen3.6 vision)
                                  ─► tags + summary ─► LanceDB ─► graph
```

- **Search** covers every passage of every note, not just the start: the query
  is matched against ~1,200-character chunks, re-scored by Qwen3-Reranker, and
  each hit shows the matching passage. Podcast hits include the timestamp and
  open the episode player at that moment.
- **Graph** edges combine semantic similarity of whole notes with shared tags.
  The graph has its own full-screen view (`/graph`): find a note, click to see
  its links, double-click to open it.
  Rejecting a suggested link removes it and keeps it out of future rebuilds.

## Environment

Runs on one machine: Windows 11, RTX 5090 (32 GB), with uv, Python 3.12,
FFmpeg, Node.js and Ollama already installed. The `.venv`, `.env`,
`frontend/.env.local`, the Ollama model and the Hugging Face models are in place.

To update Python packages (PyTorch comes from the CUDA 13 wheel index):

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[dev]" `
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
```

## Use

```powershell
.\start.ps1                 # API on http://localhost:8000, web UI on http://localhost:3000
```

Or from the CLI:

```powershell
.\zc.ps1 podcast import https://example.com/feed.xml --limit 5   # download + queue episodes (show notes help name speakers)
.\zc.ps1 podcast reprocess "Scott Frey"                           # redo analysis/corrections (or --all; --retranscribe redoes speech too)
.\zc.ps1 podcast verify                                           # compare with transcripts published in the feed
.\zc.ps1 podcast add D:\Podcasts\SomeShow                         # queue local audio
.\zc.ps1 podcast run                                              # transcribe, enhance, index
.\zc.ps1 ingest D:\Papers                                         # PDFs / markdown / images
.\zc.ps1 add https://example.com/article                          # web page
.\zc.ps1 search "how to train the gut for long races"
.\zc.ps1 stats
.\zc.ps1 graph rebuild
.\zc.ps1 reindex            # after changing EMBEDDING_MODEL or CHUNK_SIZE
```

Audio dropped into `_BRAIN_STORAGE\podcasts\<Show name>\` is picked up
automatically by the next `podcast run`.

## Storage layout

```
_BRAIN_STORAGE/
  podcasts/<show>/*.mp3      episode audio
  transcripts/*.md           transcript notes (frontmatter + summary + transcript)
  uploads/                   files added through the web UI
  .lancedb/                  notes, chunks + vectors, graph edges
  podcast_queue.json         transcription queue
  image_queue.json
```

Transcript notes are plain markdown and the source of truth: `reindex`
rebuilds the database from them.

## Configuration

Settings live in `.env`. All options and their defaults are listed in
`.env.example` and `src/zettlecast/config.py`.

## Development

```powershell
.\.venv\Scripts\python -m pytest tests
cd frontend; npm run build
```

See `ARCHITECTURE.md` for how the pieces fit together and `AGENTS.md` for the
API reference.
