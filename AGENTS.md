# Zettlecast Agent Reference

Reference for AI agents working on this repository.

## Project Overview

Local-first Zettelkasten for **Windows + NVIDIA GPU only**. Podcasts are
transcribed with speaker labels, everything is chunked and embedded into
LanceDB for semantic search, and notes are linked into a knowledge graph.

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.12 + FastAPI, CLI with Click |
| Frontend | Next.js 16 + React 19 + TypeScript |
| Database | LanceDB (notes, chunks with 1024-dim vectors, graph edges) |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` (sentence-transformers, CUDA) |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` (CrossEncoder, CUDA) |
| ASR | NeMo `nvidia/parakeet-tdt-0.6b-v2` (local attention, word timestamps) |
| Diarization | NeMo `nvidia/Nemotron-3-Diarization` (Sortformer, up to 8 speakers) |
| LLM + vision | Ollama `qwen3.6:35b-a3b` (structured JSON output, 32K context) |

### Directory Structure

```
src/zettlecast/
  main.py            FastAPI app (all endpoints)
  cli.py             Click CLI (`zettlecast ...`)
  pipeline.py        Shared orchestration: ingest_note, ingest_path, run_podcast_queue, reindex_all
  config.py          Settings (from .env at the project root)
  db.py              LanceDB tables: notes, chunks, graph_edges, rejected_edges
  embeddings.py      Embedding + reranker models
  llm.py             Ollama client (chat with optional JSON schema, images; unload)
  search.py          Chunk search -> rerank -> best chunk per note; related notes
  linker.py          Graph edges: alpha*cosine(note vectors) + beta*jaccard(tags)
  chunker.py         Prose + transcript-aware chunking (keeps "[t] speaker:" headers)
  parser.py          PDF (pypdf -> Marker), web (Trafilatura), markdown (frontmatter-aware)
  enricher.py        LLM tags + summary for non-podcast notes
  identity.py        UUID frontmatter, wikilinks
  models.py          Pydantic models
  podcast/
    nemo_transcriber.py  Parakeet + Nemotron diarization on CUDA
    aligner.py           Word -> speaker alignment, micro-segment merge
    enhancer.py          LLM analysis (keywords, summary, key points, chapters, speaker names) + cleanup
    formatter.py         Transcript markdown note (uuid = episode id)
    queue.py             Persistent transcription queue
    rss.py               Feed parsing (show notes, podcast:transcript links) + episode download
    reference.py         Publisher's transcript: speaker names by time overlap, verification report
  image/
    vision_backend.py    One structured vision call (description, OCR text, concepts)
    image_parser.py      Image -> note
    queue.py             Image queue
frontend/            Next.js UI (notes, podcasts, images, search, graph, settings)
tests/               pytest suite (LLM and models mocked)
start.ps1            Start Ollama + API + web UI
zc.ps1               CLI shortcut
```

---

## Setup & Running

```powershell
.\start.ps1                                          # API :8000 + web UI :3000
.\zc.ps1 --help                                      # CLI
```

- Single machine only; the environment (.venv, .env, models) is already installed. No setup script.
- PyTorch comes from the CUDA 13 wheel index (RTX 50-series needs torch >= 2.7):
  `uv pip install --python .venv\Scripts\python.exe -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match`
- NeMo is installed from GitHub at a pinned commit (`pyproject.toml`): the
  diarization model needs `rope` attention, which is newer than the 3.0.0 PyPI release.
  Switch the dependency to `nemo-toolkit>=3.1` once 3.1 is published (main is already 3.1.0).
- API token lives in `.env` (`API_TOKEN`) and `frontend/.env.local` (`NEXT_PUBLIC_API_TOKEN`).

### Windows gotchas (already handled in code)

- `zettlecast/__init__.py` imports torch before anything else: importing lancedb/pyarrow
  first makes torch's `c10.dll` fail (WinError 1114).
- Triton isn't available on Windows, so `nemo_transcriber.py` sets `TORCHDYNAMO_DISABLE=1`.
- Never call `torch.cuda.empty_cache()` between Parakeet `transcribe()` calls; it aborts the process.
- Parakeet must use local attention for audio longer than ~10 minutes
  (full attention fails with "CUDA driver error: device not ready").
- The LLM (~22 GB) and NeMo (~14 GB) can't share the GPU; `pipeline.run_podcast_queue`
  unloads Ollama before each transcription and NeMo frees its models before enhancement.

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `serve [--port P] [--no-frontend]` | Start API (and web UI) |
| `ingest PATH [--name SHOW]` | Ingest file/folder; audio is queued for transcription |
| `add URL` | Ingest a web page |
| `search QUERY [-k N]` | Semantic search (shows snippet + timestamp) |
| `stats` | Notes / chunks / edges counts |
| `reindex` | Re-read transcript notes from disk, re-embed, rebuild graph |
| `graph rebuild` | Recompute all edges |
| `token` | Print API token + bookmarklet |
| `podcast add PATH [--name] [--recursive]` | Queue local audio |
| `podcast import URL [--limit N]` | Download + queue feed episodes |
| `podcast run [--limit N] [--no-enhance] [--no-sync]` | Process pending episodes end to end |
| `podcast reprocess MATCH` / `--all` | Re-queue completed episodes (note is replaced in place) |
| `podcast verify [MATCH] [-s N]` | Compare notes with the feed's published transcripts (word/speaker agreement, passages to spot-check) |
| `podcast status` / `podcast retry` | Queue status / reset failed |
| `image add PATH` / `image run` / `image status` / `image retry` | Image pipeline |

`podcast run` pipeline per episode: unload LLM → decode to 16 kHz mono →
Parakeet (60-min pieces) → Nemotron diarization (whole episode) → align →
speaker names from the publisher's transcript (if it names speakers) → LLM analysis + cleanup
(given show notes, names from earlier episodes of the show and the podcast name as spellings) → `transcripts/<title>_<id8>.md` → chunk + embed → graph edges.

---

## API Endpoints

All endpoints need `?token=API_TOKEN` except `/health` and `/docs`.

```
GET  /health
POST /ingest?url=URL                    -> {status: success|duplicate|failed, uuid, title}
POST /ingest (multipart file)           -> {status: success|queued|duplicate|failed, ...}
     Files are kept in storage/uploads (audio: storage/podcasts/Uploads, then queued)
GET  /notes?status=&source_type=&limit=&offset=   -> {notes: [{uuid,title,source_type,status,created_at,tags}], count}
GET  /notes/{uuid}?include_suggestions=true      -> note + metadata + suggestions (graph neighbours)
GET  /notes/{uuid}/source               original file (audio/pdf/image), or {url} for web notes
GET  /notes/{uuid}/markdown             note as a markdown download
DELETE /notes/{uuid}
GET  /search?q=&top_k=5&rerank=true     -> {results: [{uuid,title,score,snippet,source_type,timestamp_seconds}]}
POST /notes/{uuid}/link {target_uuid, action: accept|reject}
GET  /graph?limit=2000                  -> {nodes: [{id,name,source_type,val}], links: [{source,target,value}]}
POST /graph/rebuild                     -> {edges}

GET  /podcast/status | POST /podcast/import {feed_url,limit} | POST /podcast/run {limit}
GET  /podcast/running | POST /podcast/cancel/{episode_id}
POST /podcast/retry | POST /podcast/sync | POST /podcast/reset-stuck

GET  /image/status | POST /image/scan | POST /image/add | POST /image/run {limit}
GET  /image/running | POST /image/retry

GET  /settings | POST /settings {key: value}   (written to .env; restart to apply)
```

---

## Data Model

### Transcript note (`_BRAIN_STORAGE/transcripts/*.md`)

```yaml
---
uuid: <episode id>            # stable: reprocessing replaces the note
title: Episode title
source_type: audio
source: D:/.../podcasts/Show/episode.mp3
status: inbox
created: '2026-10-16T08:00:00'
duration_seconds: 3813
language: en
speakers: 2
speaker_names: {speaker_0: Jason Koop, speaker_1: Scott Frey}
tags: [brain endurance training, mental fatigue, ...]
summary: 3-5 sentences
podcast: {show: KoopCast, episode: ..., feed_url: ..., published: ...}
---

## Summary
## Key Points
## Chapters
## Show Notes           (from the RSS feed, if imported from one)
## Needs Review          (low-confidence LLM corrections, if any)
## Transcript
[8.1s - 97.6s] Jason Koop: ...
[97.8s - 103.0s] Scott Frey: ...
```

### LanceDB tables

- **notes**: note fields + `metadata_json` + `vector` (normalised mean of chunk vectors)
- **chunks**: `chunk_id`, `note_uuid`, `text`, `start_char`, `vector`
- **graph_edges**: `source_uuid` (older) → `target_uuid` (newer), `weight`, `vector_sim`, `tag_sim`
- **rejected_edges**: pairs never to link

---

## Testing

```powershell
.\.venv\Scripts\python -m pytest tests
cd frontend; npm run build
```
