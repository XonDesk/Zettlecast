# Zettlecast Agent Reference

Reference for AI agents working on this repository.

## Project Overview

Local-first Zettelkasten for **Windows + NVIDIA GPU only**. Podcasts are
transcribed with speaker labels, everything is chunked and embedded into
LanceDB for semantic search, and notes are linked into a knowledge graph.
It can also run split across two machines (`ROLE`, see "Two machines" below): an always-on Mac
mini as the primary (API, web UI, database) and the GPU PC as a worker it wakes for jobs.

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
  reddit.py          Reddit saves -> notes: private saved feed (RSS), thread feeds, paced client
  insights.py        Today's cards: surprising graph links + echoes (new vs much older), picked ahead; LLM says what
                     links them (agree/contradict/extend/apply/unrelated); verdicts; notes written from a card
  brief.py           A note at a glance (summary / image description / start; has_image; iso_date): graph + cards
  curation.py        Reddit comment curation: the user's rubric (<storage>/reddit_curation.md, Settings page) -> takeaways
  runstate.py        Run state shared across processes (<storage>/run_state.json) + stop requests; one run at a time;
                     cycle times per phase, step times, per-phase averages carried to the next run (HMI ETA)
  system.py          Process page status (run, queues, GPU, LLM server, DB, disk, alarms) and controls
  statefile.py       JSON state with merge-on-save (queues: writers never undo each other's changes)
  paths.py           "desktop:<path>" references to files on the worker
  worker_api.py      Primary: /worker/* (heartbeat, claim, inputs, results, link), WORKER_TOKEN
  workerstate.py     Primary: the worker's last heartbeat, commands for it, pause / stay on (worker_state.json)
  wake.py            Primary: Wake-on-LAN + the policy (jobs waiting, worker offline, WAKE_WINDOW, give up after 3)
  tagging.py         LLM tags the primary couldn't do: found in the DB (no summary / uncurated save), done by the worker
  worker.py          Worker: claim -> fetch inputs -> pipeline -> send results; heartbeat; idle shutdown
  worker_files.py    Worker: file server for the primary (audio with Range, screenshot triage)
  migrate.py         `migrate-paths`: desktop paths in copied data -> the primary's copies or desktop: refs
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
frontend/            Next.js UI: app/(main)/ pages share the sidebar (notes, podcasts, images,
                     process, settings; app/(main)/search still exists but is hidden: the graph's search box replaces it);
                     app/(main)/process is the HMI-style control page (ISA-101 look; cycle times + rough ETA; the image line
                     screenshots -> sorting -> vision with its controls); app/graph is a full-screen view;
                     app/triage sorts screenshots (swipe / arrow keys; reached from Images and the process page); components/Transcript renders speaker turns
tests/               pytest suite (LLM and models mocked)
start.py             Start Ollama (if needed) + API + web UI (any OS)
zc.py                CLI shortcut (any OS)
```

---

## Setup & Running

```sh
python start.py                                          # API :8000 + web UI :3000
python zc.py --help                                      # CLI
```

- On this PC the environment (.venv, .env, models) is already installed. The two-machine setup has
  its own guides: `deploy/macos/README.md` (primary) and `deploy/windows/README.md` (worker).
- PyTorch comes from the CUDA 13 wheel index (RTX 50-series needs torch >= 2.7); NeMo, torchaudio
  and Marker are the `[gpu]` extra (the Mac primary installs without it):
  `uv pip install --python .venv\Scripts\python.exe -e ".[gpu,dev]" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match`
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
| `worker [--once] [--no-file-server]` | Worker (`ROLE=worker`): do the primary's jobs; started at boot by Task Scheduler |
| `wake` | Primary: Wake-on-LAN to the worker now |
| `migrate-paths --desktop-root PATH [--dry-run]` | Primary, once after copying the storage folder: repoint paths |
| `reddit sync [--limit N] [--refresh] [--no-articles] [--no-llm] [--no-threads]` / `reddit tag` / `reddit status` | Import Reddit saves (`--refresh` fetches threads missing so far; `tag` adds LLM tags to saves imported without) |

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
GET  /notes/{uuid}/thumb                small JPEG of a screenshot kept with its image (<storage>/thumbs)
DELETE /notes/{uuid}
GET  /search?q=&top_k=5&rerank=true     -> {results: [{uuid,title,score,snippet,source_type,timestamp_seconds}]}
POST /notes/{uuid}/link {target_uuid, action: accept|reject}
GET  /graph?limit=2000                  -> {nodes: [{id,name,source_type,val,community,brief,image,tags,people,organisations,source,date,duration}],
                                            links: [{source,target,value,kind}],
                                            communities: [{id,label,size}]}
POST /graph/rebuild                     -> {edges}   (edges + communities)

GET  /today                             today's insight cards (planned on the first call of the day; standalone
                                        starts the LLM on unprepared ones in the background)
POST /today/{key}/verdict {verdict: good|rejected}   rejected = the pair is unlinked for good
POST /today/{key}/note {title, text}    a note of your own in <storage>/notes, [[linked]] to both

GET  /podcast/status | POST /podcast/import {feed_url,limit} | POST /podcast/run {limit}
GET  /podcast/running | POST /podcast/cancel/{episode_id}
POST /podcast/retry | POST /podcast/sync | POST /podcast/reset-stuck

GET  /image/status | POST /image/scan | POST /image/add | POST /image/run {limit}
GET  /image/running | POST /image/retry
POST /image/stop                        finish the images being read, start no more, link what was stored
GET  /image/triage?limit=               next undecided screenshots (newest first), counts, folders, queue
POST|DELETE /image/triage/folders {path} | GET /image/triage/file?path= (only inside a triage folder)
POST /image/triage/decide {path, decision: keep|keep_image|skip} | POST /image/triage/undo {path}
POST /image/triage/process              read every kept screenshot (vision model; refused during a podcast run)

GET  /system/status                     run state (any process), queues, GPU, LLM server, Ollama, DB, disk, alarms
POST /system/run/stop {mode: after_episode|abort} | POST /system/run/resume
POST /system/unload (free the GPU; not during a run) | POST /system/reindex (background task)

GET  /reddit/status                     feed set?, imported, without thread, last sync, sync progress
POST /reddit/sync?refresh=&enrich=&threads=   background task (refused during a podcast run)
POST /reddit/tag | POST /reddit/stop     tag untagged saves (LLM, no Reddit requests) | stop after the current save

GET  /settings | POST /settings {key: value}   (written to .env; restart to apply)

POST /system/wake | POST /system/desktop {stay_on}     primary: wake the worker / keep it on when idle

# Primary only, Bearer WORKER_TOKEN (not the API token): the worker's side of the job queues
POST /worker/heartbeat {worker_id, busy, run, images, task, gpu, llm, triage, ...} -> commands
POST /worker/claim {worker_id, kinds, limits}  -> {kind: podcast|image|tag|null, items}
GET  /worker/input/{kind}/{job} | POST /worker/{kind}/{job}/fetched {ref}
POST /worker/podcast/{job}/started|completed|failed | POST /worker/podcast/release
POST /worker/image/{job}/completed|failed | POST /worker/image/release|add|remove
POST /worker/tag/{uuid} | POST /worker/tag/release | POST /worker/link
POST /worker/insight/{key} {insight|error} | POST /worker/insight/release {keys}
```

### Two machines (ROLE=primary / ROLE=worker)

`ROLE=standalone` (default) is everything on one PC, as above. Split:

- **Primary** (Mac mini, always on): API (0.0.0.0, for the worker), web UI, LanceDB, notes, both queues,
  Reddit sync, graph rebuilds, search (embedder/reranker on MPS float16). It never loads an LLM
  (`llm.chat` raises, `enrich_note` returns the note as is): notes are stored at once and
  `tagging.py` finds the ones still without summary for the worker. GPU endpoints (`/podcast/run`,
  `/image/run`, `/image/triage/process`, `/reddit/tag`) unpause the worker and wake it. Triage and a
  `desktop:` note source are forwarded to the worker's file server (503 "desktop is off" when it isn't
  heard from in 30 s). Process page: `system.status()` uses the worker's heartbeat; Stop pauses the
  worker (no new claims until Resume).
- **Worker** (GPU PC, normally off): `zettlecast worker` claims one kind per batch (podcasts first,
  10 at a time, so the two passes still load the LLM once; then images; then tags), downloads inputs
  the primary holds (uploads, feed downloads) into its own storage and tells the primary to use that
  copy (`desktop:...`; the primary deletes its own), runs the same pipeline with `RemoteTranscriptionQueue`
  / `RemoteImageQueue` (results posted to the primary, which stores, embeds and links them; one `/worker/link`
  rebuild per batch). Keeps the audio, `.asr.json`, glossaries, lookups and its own transcript copies (known
  speakers). Claims are **leases** renewed by the heartbeat; a worker id is new per start, so a crashed
  worker's jobs go back to pending after `WORKER_LEASE_MINUTES`. Idle `WORKER_IDLE_SHUTDOWN_MINUTES` with no
  Windows session signed in (WTS API; console or RDP) and not "stay on": `shutdown /s /t 120` (cancelled if
  someone signs in).
- Queue files are written by several threads on the primary: `statefile.save` merges three ways (as
  loaded / now / on disk), so a long feed import doesn't undo the worker's results.

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
↑ keep with the image (copied to `_BRAIN_STORAGE/images/`, shown on the note). The Discard button
(no key or swipe) adds nothing and leaves the file where it is. Kept files
are never changed. Decisions are keyed by content hash (`_BRAIN_STORAGE/image_triage.json`), so
moved or renamed screenshots aren't asked about again. The vision call finds the screenshot's
**focus** (the post / comment it was taken for: highlighted, else fully in view near the middle)
and its context (site, author, what it replies to); only those, the description and concepts
are embedded. The call also says what the screenshot was **kept as** (`saved_as`: media, quote, idea,
tip, place, product, recipe, reference, other) and the thing itself (`item`: name, by, type, where,
price): it leads the note (`## Kept`, indexed), the title ("Book: Dune", "Place: Kamikochi") and the
tags ("to read", "to watch", "to listen", "to play", "place to visit", "to buy", "quote"...;
`formatter.saved_tag`). Image note uuids come from the image's bytes, so reading one again replaces its note. An image run
reads `LLM_PARALLEL` images at once and links the new notes with one graph rebuild at the end
(linking per note rebuilds the graph each time: ~10 s of CPU against ~2 s to read an image). **Circling** (✎ / M on the triage page) saves the stroke and its box (fractions of
the image) with the decision; the vision call then gets the screenshot with the mark drawn on it
plus a padded, enlarged crop of the circle, and reads the circled content as the focus (it
overrides the model's own pick). The rest of the screen goes in `## Screen Text` and file details in
`## Image Details` (both in `chunker.UNINDEXED_SECTIONS`). People / organisations must appear in
the focus or context (ads and neighbouring comments are dropped).

### Reddit saves

`REDDIT_SAVED_FEED` (Settings page; read from .env at each sync, no restart) is the private RSS link
for "your saved links" on reddit.com/prefs/feeds: no Reddit app or login, newest ~1000 saves.
Reddit sends anonymous JSON to its login page, so everything is read as RSS (Atom): one thread
feed per save (`depth=1`, top order; no scores or reply nesting in feeds). Measured Oct 2026: about
**one request a minute** anonymously (429 with `x-ratelimit-remaining: 0`); `reddit.Client` paces itself
(6.5 s, or Reddit's rate-limit headers) and merges its params into the link's own query (httpx's `params=`
replaces it: the feed key was dropped -> 404). OAuth is no way out: since Nov 2025 new Reddit API apps
need manual approval. Hence **quick sync** (`threads=False`: notes from the saved feed alone, ~100 a
minute) then **fetch comments** (`refresh=True`, ~1 a minute, resumable, keeps tags);
`enrich=False` skips the LLM (GPU busy) and `tag_untagged` / `reddit tag` / `POST /reddit/tag` adds
tags + summaries later from the database (no Reddit requests); `POST /reddit/stop` ends a sync after the
current save. A sync stores notes with `db.upsert_note` and links them with **one** `rebuild_graph()` at
the end (`update_edges_for_note` is ~10 s per note at ~1300 notes). One note per save,
`source_type: reddit`, uuid from the Reddit id (refresh replaces in place): a post has its text, the
linked article (web parser, `parse_url(enrich=False)`; media and Reddit links skipped) and the top
comments; a saved comment has the post and the other top comments (up to 25 top-level, raw in
`custom.reddit.comments` and the unindexed `## Thread comments`). **Curation** (`curation.py`, before tagging):
the LLM judges every comment by the user's rubric (products, advice, experience, insight kept; jokes, banter,
debate dropped; a saved comment is always kept) and writes one-line takeaways to `## From the comments`,
which is indexed; tags come from that, not the raw comments. `reddit tag` also curates stored notes (older
ones have `## Top comments`, converted); `reddit review --out f.json` curates a sample without storing.
LLM tags/summary as for web notes. Graph source group per subreddit (`reddit:r/<sub>`), date = posted. Imported ids in
`_BRAIN_STORAGE/reddit_saved.json`; a note deleted in the UI isn't imported again.

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
Fill = community: picked per graph so linked communities (neighbours on the map) differ most
(`autoCommunityColors`: 12 hues, then golden-angle ones; any changeable in the pane, Reset colours drops
the picks; communities also named on the map), ring = source type. Node sizes are in layout units (`itemSizesReference: 'positions'`), so the collision
radius matches what's drawn and titles appear as you zoom in. The right-hand pane
(`GraphPane.tsx`, settings in localStorage) has Filters (query language in `lib/graphQuery.ts`:
`tag:` `person:` `org:` `source:` `type:` `community:` `year:2021-2023`, `-` negates; types, years,
link kinds, minimum strength, orphans, neighbourhood of the selected note), Communities (show/hide,
highlight, what stands out in each: topics/people/organisations at 1.5x their overall rate),
Groups (colour by community/year/source/type, query -> colour groups), Display and Forces.
The search box matches titles as you type and, after a pause, runs passage search (`/search`, embeddings +
reranker): the notes found are highlighted on the map, each result opens the note at the passage's time.
Display → Density contours (off by default) draws Embedding Atlas-style KDE outlines per colour group
under the links (`components/graphContours.ts`, d3-contour; computed in layout units, redrawn per frame).

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
