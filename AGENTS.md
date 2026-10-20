# Zettlecast Agent Reference

Reference for AI agents working on this repository.

## Project Overview

Local-first Zettelkasten for **Windows + NVIDIA GPU only**. Podcasts are
transcribed with speaker labels, everything is chunked and embedded into
LanceDB for semantic search, and notes are linked into a knowledge graph.

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.12 + FastAPI, CLI with Click |
| Frontend | Next.js 16 + React 19 + TypeScript; graph: Sigma.js (WebGL) + d3-force |
| Database | LanceDB (notes, chunks with 1024-dim vectors, graph edges) |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` (sentence-transformers, CUDA) |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` (CrossEncoder, CUDA) |
| ASR | NeMo `nvidia/parakeet-tdt-0.6b-v2` (local attention, word timestamps) |
| Diarization | NeMo `nvidia/Nemotron-3-Diarization` (Sortformer, up to 8 speakers) |
| LLM + vision | `qwen3.6:35b-a3b` served by llama.cpp `llama-server` (Ollama's GGUF; parallel slots, speculative decoding), Ollama as fallback |

### Directory Structure

```
src/zettlecast/
  main.py            FastAPI app (all endpoints)
  cli.py             Click CLI (`zettlecast ...`)
  pipeline.py        Shared orchestration: ingest_note, ingest_path, run_podcast_queue, reindex_all
  config.py          Settings (from .env at the project root)
  db.py              LanceDB tables: notes, chunks, graph_edges, rejected_edges, communities
  embeddings.py      Embedding + reranker models
  llm.py             LLM client: llama-server (started on demand) or Ollama; JSON schema, images, chat_each, unload
  search.py          Chunk search (max 5 chunks per note) -> rerank -> best chunk per note; related notes
  linker.py          Graph edges (centred similarity + matching passages + rare shared entities, k best per note) + Leiden communities
  chunker.py         Prose + transcript-aware chunking (keeps "[t] speaker:" headers)
  parser.py          PDF (pypdf -> Marker), web (Trafilatura), markdown (frontmatter-aware)
  enricher.py        LLM tags + summary for non-podcast notes
  identity.py        UUID frontmatter, wikilinks
  models.py          Pydantic models
  runstate.py        Run state shared across processes (<storage>/run_state.json) + stop requests; one run at a time
  system.py          Process page status (run, queues, GPU, LLM server, DB, disk, alarms) and controls
  podcast/
    nemo_transcriber.py  Parakeet + Nemotron diarization on CUDA
    asr_cache.py         Saved raw transcription (<audio>.asr.json) between the two passes
    aligner.py           Word -> speaker alignment, micro-segment merge
    enhancer.py          LLM analysis (keywords, summary, key points, chapters, speaker names) + cleanup
    formatter.py         Transcript markdown note (uuid = episode id)
    queue.py             Persistent transcription queue
    rss.py               Feed parsing (show notes, podcast:transcript links) + episode download
    reference.py         Publisher's transcript: speaker names by time overlap, verification report
    acoustic.py          Parakeet as acoustic judge: NLL of a text for an audio clip (vectorised TDT forward)
    verifier.py          Audio-checked corrections: suspects/publisher/glossary/lookup proposals, cleanup edits, glossary learning
    lookup.py            Online name lookup (Wikipedia): real names found via what the passage says about them
  image/
    vision_backend.py    One structured vision call (screenshot focus + context, description, OCR text, concepts)
    image_parser.py      Image -> note
    queue.py             Image queue
    triage.py            Screenshot triage: folders, keep / keep_image / skip decisions by content hash
frontend/            Next.js UI: app/(main)/ pages share the sidebar (notes, podcasts, images, search,
                     process, settings); app/(main)/process is the HMI-style control page (ISA-101 look); app/graph is a full-screen view;
                     app/triage sorts screenshots (swipe / arrow keys); components/Transcript renders speaker turns
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
- API token lives in `.env` (`API_TOKEN`) and, server-side only, in `frontend/.env.local` (`ZC_API_TOKEN`).
- **Web UI sign-in** (`frontend/proxy.ts`, `lib/auth.ts`): every page needs a session cookie (signed,
  HttpOnly, 1 year); `/login` posts to `/auth/login` (PBKDF2 hash in `ZC_LOGIN_HASH`, user
  `ZC_LOGIN_USER`, case-insensitive; slows and locks after repeated failures). The browser calls
  `/zc-api/<path>`; the gate forwards it to the API on 127.0.0.1 with the token, so the token is
  never in the JavaScript. `WEB_LAN=true` (`.env`) makes `serve` start the web UI on 0.0.0.0 for
  phones on the home network; the API stays on 127.0.0.1. Windows Firewall must allow Node on
  private networks (the user's call).

### Windows gotchas (already handled in code)

- `zettlecast/__init__.py` imports torch before anything else: importing lancedb/pyarrow
  first makes torch's `c10.dll` fail (WinError 1114).
- Triton isn't available on Windows, so `nemo_transcriber.py` sets `TORCHDYNAMO_DISABLE=1`.
- Never call `torch.cuda.empty_cache()` between Parakeet `transcribe()` calls; it aborts the process.
- Parakeet must use local attention for audio longer than ~10 minutes
  (full attention fails with "CUDA driver error: device not ready").
- The LLM (~24 GB in llama-server) and NeMo (~14 GB) can't share the GPU; `pipeline.run_podcast_queue`
  stops llama-server / unloads Ollama before the transcription pass and NeMo frees its models before the analysis pass.
  A second, small Parakeet (fp16, ~1.3 GB) stays loaded as the acoustic judge next to the LLM.
- The API server unloads the search models (~3 GB) after `MODEL_IDLE_MINUTES` (5) without a search
  (`embeddings.start_idle_unloader`), so a CLI transcription run isn't starved of GPU memory
  (an episode took 23 min instead of 3.5 with them loaded). `run_podcast_queue` holds
  `embeddings.hold()` so the cache is never emptied mid-run.
- NeMo's TDT loss kernel needs NVVM (CUDA toolkit, not installed), so `acoustic.py`
  computes the TDT likelihood itself; it matches `TDTLossPytorch` exactly and is ~600x faster.
- **LLM backend** (`LLM_BACKEND=llama-server`, default): `llm.py` starts llama.cpp's `llama-server.exe`
  (newest `%LOCALAPPDATA%\llama.cpp\<build>\`, currently b11399 CUDA 13.4) on the GGUF in Ollama's store
  (`~/.ollama/models`, found via the manifest; no second copy), 4 slots sharing one 64K context
  (`--kv-unified`), `--spec-type ngram-simple,draft-mtp`: n-gram drafts copy the transcript being
  cleaned (~95% accepted), MTP uses the model's own draft layer. Cleanup chunks ~2.2 s vs 8.3 s on
  Ollama; pass 2 ~1.0-1.3 s per audio minute vs 2.1-2.5. The vision projector is loaded only for image
  calls (server restarted with `--mmproj`). Stopped by `unload()` (pass 1), after `LLM_IDLE_MINUTES`
  unused, and at exit of the process that started it (pid in `<storage>/.llama-server.pid`, log in
  `<storage>/llama-server.log`). No llama-server.exe -> Ollama. Ollama 0.40.6 can't batch qwen35moe.
- **VRAM budget in pass 2** (32 GB card): llama-server ~24 GB + acoustic judge 1.2 GB + embedder 1.2 GB
  + desktop 0.8 GB. Past 32 GB Windows silently spills to system RAM and the LLM drops from ~300 to
  ~80 tokens/s, so: the judge loads on the CPU and is halved there (NeMo loads float32 onto the GPU),
  `tdt_log_likelihood` takes the log-softmax normaliser blockwise (a float64 copy of the joint tensor
  was GBs), embedding batch 8 (half the peak of 32), and pass 2 empties PyTorch's cache per episode.
- Queue runs hold a Windows power request (`pipeline.keep_awake`) so Modern Standby doesn't pause them.

### Transcript verification (no human needed)

Per episode: `## Automatic Corrections` lists applied fixes (time, source, confidence,
acoustic penalty); `## Needs Review` lists name-like spots where a plausible alternative
had about as much support; frontmatter `verification` has the counts. The show glossary
(`_BRAIN_STORAGE/glossaries/<show>.json`, term -> evidence) grows per episode; terms count
as known at 2+ (confirmed speakers, publisher-agreed corrections and LLM names confirmed online
add 2; LLM-only or lookup-only 1). Names nothing local confirms are looked up on Wikipedia
(`NAME_LOOKUP`, the only online step; cached in `_BRAIN_STORAGE/lookups/wikipedia.json`): the
suspects pass says who the person is ("Norwegian triathlete"); result pages and the pages they
link to confirm exact spellings (+2.5, beats a confident LLM guess), and only a result page with
the same word count and a close spelling is proposed as a new name. Single words never are.
`podcast verify` measures agreement with the publisher's transcript.

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
| `graph rebuild` | Recompute all edges and communities |
| `graph stats` / `graph explain A B` | Links per note, communities; why two notes are (not) linked |
| `token` | Print API token + bookmarklet |
| `podcast add PATH [--name] [--recursive]` | Queue local audio |
| `podcast import URL [--limit N]` | Download + queue feed episodes |
| `podcast run [--limit N] [--no-enhance] [--no-sync]` | Process pending episodes end to end |
| `podcast reprocess MATCH` / `--all` [`--retranscribe`] | Re-queue completed episodes (note is replaced in place); reuses the saved transcription unless `--retranscribe` |
| `podcast verify [MATCH] [-s N]` | Compare notes with the feed's published transcripts (word/speaker agreement, passages to spot-check) |
| `podcast status` / `podcast retry` | Queue status / reset failed |
| `image add PATH` / `image run` / `image status` / `image retry` | Image pipeline |

`podcast run` works in two passes, so the LLM is loaded once per run. Pass 1, for every
pending episode (speech models reloaded per episode: a Parakeet kept across a diarization
crashes with an illegal memory access): decode to 16 kHz mono → Parakeet (60-min pieces) → Nemotron diarization (whole
episode) → align → save `<audio>.asr.json` (reused by later runs until the models change).
Pass 2, with the LLM and acoustic judge loaded once: glossary seeded with every episode's
publisher-confirmed speakers → per episode:
speaker names from the publisher's transcript (if it names speakers) → LLM analysis →
suspects pass (LLM lists likely misheard names/terms with sound-alike candidates; publisher
differences, the show glossary and a Wikipedia lookup for unconfirmed names add candidates) → each candidate scored against the audio
(rejected above 10 nats per changed word, 30 for a confirmed name respelled; best-supported plausible candidate wins) →
LLM cleanup, every content edit also checked against the audio → `transcripts/<title>_<id8>.md` → chunk + embed → graph edges.

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
GET  /graph?limit=2000                  -> {nodes: [{id,name,source_type,val,community,tags,people,organisations,source,date,duration}],
                                            links: [{source,target,value,kind}],
                                            communities: [{id,label,size}]}
POST /graph/rebuild                     -> {edges}   (edges + communities)

GET  /podcast/status | POST /podcast/import {feed_url,limit} | POST /podcast/run {limit}
GET  /podcast/running | POST /podcast/cancel/{episode_id}
POST /podcast/retry | POST /podcast/sync | POST /podcast/reset-stuck

GET  /image/status | POST /image/scan | POST /image/add | POST /image/run {limit}
GET  /image/running | POST /image/retry
GET  /image/triage?limit=               next undecided screenshots (newest first), counts, folders, queue
POST|DELETE /image/triage/folders {path} | GET /image/triage/file?path= (only inside a triage folder)
POST /image/triage/decide {path, decision: keep|keep_image|skip} | POST /image/triage/undo {path}
POST /image/triage/process              read every kept screenshot (vision model; refused during a podcast run)

GET  /system/status                     run state (any process), queues, GPU, LLM server, Ollama, DB, disk, alarms
POST /system/run/stop {mode: after_episode|abort} | POST /system/run/resume
POST /system/unload (free the GPU; not during a run) | POST /system/reindex (background task)

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
tags: [brain endurance training, mental fatigue, ...]   # topics only, no names
people: [Scott Frey, Jason Koop]                        # full names, snapped to known spellings
organisations: [UESCA, CTS Coaching]                    # organisations, products, races, events
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

### Screenshots

Triage (`/triage` page): ← delete (nothing added; the file goes to the Windows Recycle Bin
`triage.RECYCLE_DELAY` = 10 min later, by a sweeper thread in the API server, so Undo until then
just cancels it; only the very file decided on, checked by key or hash), → keep (the information only: the note
records the original path, `image_kept: false`, and `/notes/{uuid}/source` won't serve it),
↑ keep with the image (copied to `_BRAIN_STORAGE/images/`, shown on the note). Kept files
are never changed. Decisions are keyed by content hash (`_BRAIN_STORAGE/image_triage.json`), so
moved or renamed screenshots aren't asked about again. The vision call finds the screenshot's
**focus** (the post / comment it was taken for: highlighted, else fully in view near the middle)
and its context (site, author, what it replies to); only those, the description and concepts
are embedded. **Circling** (✎ / M on the triage page) saves the stroke and its box (fractions of
the image) with the decision; the vision call then gets the screenshot with the mark drawn on it
plus a padded, enlarged crop of the circle, and reads the circled content as the focus (it
overrides the model's own pick). The rest of the screen goes in `## Screen Text` and file details in
`## Image Details` (both in `chunker.UNINDEXED_SECTIONS`). People / organisations must appear in
the focus or context (ads and neighbouring comments are dropped).

### Knowledge graph

Notes are the only nodes. Edge weight = 0.5 x cosine of note vectors after subtracting a *balanced*
mean (each source group - podcast show, website, images - counts once) + 0.3 x share of closely
matching passages (passage vectors centred the same way; passages repeated across a group skipped)
+ 0.2 x IDF-weighted shared topics / people / organisations (the host, in every episode, adds ~0).
Each note keeps its 6 best partners (`GRAPH_MAX_EDGES_PER_NOTE`), edges are the union; wikilinks are
always edges; rejected pairs never. Communities: Leiden (`igraph` + `leidenalg`), `GRAPH_RESOLUTION`.
Edges for a new note are computed at ingest; the full rebuild (communities) runs after each podcast
run, on `reindex`, `graph rebuild` and the UI button. The verification sections of a transcript
note (`chunker.UNINDEXED_SECTIONS`) are not chunked or embedded.

Graph view (`frontend/components/Graph.tsx`, Obsidian-like): a live d3-force simulation (centre, repel,
link force/distance, collision, plus a community pull towards each community's centre; links across
communities weaken as it grows), laid out 300 ticks before the first frame; notes can be dragged.
Fill = community (12 hues, any changeable in the pane; communities also named on the map), ring =
source type. Node sizes are in layout units (`itemSizesReference: 'positions'`), so the collision
radius matches what's drawn and titles appear as you zoom in. The right-hand pane
(`GraphPane.tsx`, settings in localStorage) has Filters (query language in `lib/graphQuery.ts`:
`tag:` `person:` `org:` `source:` `type:` `community:` `year:2021-2023`, `-` negates; types, years,
link kinds, minimum strength, orphans, neighbourhood of the selected note), Communities (show/hide,
highlight, what stands out in each: topics/people/organisations at 1.5x their overall rate),
Groups (colour by community/year/source/type, query -> colour groups), Display and Forces.

### LanceDB tables

- **notes**: note fields + `metadata_json` + `vector` (normalised mean of chunk vectors)
- **chunks**: `chunk_id`, `note_uuid`, `text`, `start_char`, `vector`
- **graph_edges**: `source_uuid` (older) → `target_uuid` (newer), `weight`, `kind` (similar | duplicate |
  link = a [[wikilink]]), `vector_sim`, `passage_sim`, `entity_sim`, `reason_json` (best passage pair with
  times, distinctive shared entities). Derived data: an older layout is dropped and refilled by a rebuild.
- **communities**: `community` (id kept stable across rebuilds), `label` (top topics), `members`
- **rejected_edges**: pairs never to link

---

## Testing

```powershell
.\.venv\Scripts\python -m pytest tests
cd frontend; npm run build
```

Fix-pass bench (real models, ~3 min per run instead of ~20): the first 10 minutes of each
processed episode in `_BENCH_STORAGE` (real notes untouched), same glossary snapshot every run.

```powershell
.\.venv\Scripts\python scriptsench.py setup   # cut clips + transcribe once (~5 min)
.\.venv\Scripts\python scriptsench.py run     # fix pass + score corrections and final transcripts (key spellings, wrong forms, WER), diff vs last run
```
