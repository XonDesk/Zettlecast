# Plan: a light primary (Mac only stores, serves and coordinates)

Status (2026-11-04): **steps 1-3 and 10 done** (the Mac runs in containers since 12:18). Next: step 4. Branch: `mac-primary-setup` (then one branch per step or one
feature branch, see Deploying).

## Goal

- You never wait to *use* Zettlecast: notes, graph, Today, search all answer from the Mac with the PC off.
- New material waits for the PC: it is queued on the Mac and becomes searchable after the PC's next session.
- The Mac does no heavy work: no LLM (already), **no embedding of notes, no graph computation, no reranking,
  no parsing**. It writes rows the PC computed, keeps the queues, wakes the PC, and hosts the web UI.

Decisions (user, 2026-11-04):

1. Wake the PC when WAKE is pressed or **5 or more** items are queued.
2. Search always on the Mac, vector search only (no reranker).
3. Reddit fetching stays on the Mac (network only); tagging and embedding on the PC.

## Measured (2026-11-04, 1743 notes, 32k chunks)

| | Mac (M1, 16 GB) | PC (285K, 5090) |
|---|---|---|
| Full graph rebuild, compute only | ~21 s | ~20 s |
| Search: load embedder / reranker | 9 s / **71 s** | |
| Search: embed + kNN | ~0.1 s | |
| Search with rerank | **~17 s** | |
| Search: query embed on CPU only (as in a Linux container) | ~250 ms, 1.4 GB | |

The graph is CPU work (O(chunks²) passage similarity), as fast on either machine; it moves anyway (goal), and
runs once per PC session instead of after every batch, which also ends the GPU idling ~30 s per batch on
`/worker/link`.

## Heavy work on the Mac today

| Where | What | Moves to |
|---|---|---|
| `db.upsert_note` (db.py:155) | chunk + embed every stored note | PC sends chunks + vectors (step 3) |
| `worker_api` podcast/image/tag results | `upsert_note` + `store_transcript` -> `update_edges_for_note` | vectors from the PC; graph job (3, 6) |
| `pipeline.ingest_note` (web/PDF/upload adds, main.py:272/297, references.py:264, insights.write_note) | parse + embed + incremental edges | ingest queue (5) |
| `reddit.py:422/526` sync | embed every save + `rebuild_graph()` at the end | stored queued, indexed by the PC (4) |
| `/worker/link` -> `linker.rebuild_graph` | full rebuild after every batch | graph job on the PC, once per session (6) |
| `search.search` (search.py:52) | reranker | off on the primary (2) |

Stays: query embedding (Qwen3-Embedding, ~1.2 GB, must be the model that made the stored vectors), kNN,
Today's card picking (`insights.candidates`, a light pass over stored edges), queues, Reddit feed fetching,
thumbnails (small Pillow resizes on view), the process page.

## Steps

Each step is one commit with its tests, and leaves both machines working. Primary and worker are updated
together from step 3 on (new payloads); the worker sends `kinds` it knows in `/worker/claim`, so an old worker
never gets a job kind it can't do.

### 1. Stop the churn (small, first)

- `reddit.llm_pass` / `tagging.llm_pass`: a save below `MIN_TEXT` (title-only image/link posts) skips the LLM
  and is returned as done (stored untagged-by-LLM, still embedded) instead of failing.
- `worker.run_tags`: only real LLM errors (server down / timeout) count toward "3 in a row = LLM down"; a
  "no usable answer" for one note doesn't release the batch.
- Clear the 46 entries in `tag_failures.json` that this explains (they get done by the new rule).
- Tests: `test_reddit.py`, `test_worker.py` (a batch with title-only saves isn't cut short).

### 2. Search on the Mac without reranking

- Setting `RERANK` (default true; false implied on `ROLE=primary`); `search.search` skips `rerank()`;
  `embeddings.get_reranker` never loads on the primary.
- Embedder stays loaded on the primary (`MODEL_IDLE_MINUTES=0` in the Mac's `.env`): no 9 s first search.
  ~1.2 GB; the Mac has ~10 GB free.
- Tests: `test_search.py` with rerank off; the reranker mock is never called on a primary.

### 3. The PC sends notes with their vectors

- `db.upsert_note(note, chunks=None, vectors=None)`: given, stored as is (no embedding). On the primary,
  without vectors it doesn't embed: it stores a **queued** row (step 4).
- Worker helper `indexed(note)`: `create_chunks` + `embed_documents` on CUDA -> `{chunks: [{chunk_id, text,
  start_char}], vectors: <float32, base64>}` (~4 KB per chunk; an hour-long episode ~100 chunks = ~0.4 MB).
- Used by: podcast completed (`store_transcript` takes the vectors; no `update_edges_for_note`), image
  completed, tag results.
- Tests: `test_worker.py`: stored vectors equal what the worker sent; the embedder mock is never called on
  the primary.

### 4. Queued notes and an "index" job

- A note can be stored on the Mac before it's indexed: `status="queued"`, no chunks, zero vector. It shows in
  the notes list (with a "waiting for the desktop" badge) and is skipped by search (no chunks), the graph,
  Today and stats counts that mean indexed notes.
- The worker's `tag` job becomes `index`: everything queued or still needing the LLM pass (`tagging._untagged`
  + queued rows): LLM pass if needed, then `indexed(note)`, back to the Mac.
- Reddit sync (`reddit.py:422/526`) stores saves queued; no `rebuild_graph()` at the end.
- `insights.write_note` (note from a Today card) stores the markdown file + a queued row.
- Audit every reader of `db.notes` for queued rows: `all_note_vectors`, `insights._notes`, `tagging._untagged`,
  `system` stats, `brief`, notes list / note view, `migrate`.
- Tests: queued rows invisible to search/graph/insights, visible in the list; index job round trip.

### 5. Adding pages and files: the ingest queue

- New `ingest_queue.json` (statefile, leases like the image queue): `{id, kind: url|file|reference, url or
  path, referenced_by, title, status}`. `/ingest` (URL or upload), `fetch_reference`, return at once with the
  queued item and a placeholder note (title = URL or file name).
- Worker job `ingest`: URL -> `parse_url` on the PC (Trafilatura; Marker fallback available there); file ->
  fetched with the existing `/worker/input` -> `parse_file`; then enrich + `indexed`, back to the Mac, which
  replaces the placeholder.
- Uploaded PDFs/markdown **stay on the Mac** (small, viewable with the PC off); audio and images keep the
  current behaviour (moved to the PC).
- Process page: "N pages/files waiting" in the Desktop panel; the existing web ETA uses the queue.
- Tests: queue round trip, placeholder replaced, failure keeps the placeholder with the error.

### 6. The graph on the PC, once per session

- Split `linker.rebuild_graph` into `compute_graph(notes, chunks, rejected, previous_communities) -> (edges,
  communities)` (pure, no DB) and `store_graph(edges, communities)` (Mac: `clear_edges`, `add_edges`,
  `replace_communities`, `optimize`).
- A "graph dirty" flag in `worker_state.json`, set whenever the Mac stores an indexed note or a link is
  added. New job kind `graph`, claimed **last** (only when no other kind has work), so one per session.
- `/worker/graph/input`: notes + chunks (vectors, texts, metadata) as Arrow IPC (~130 MB vectors + ~40 MB
  text today, a second or two on gigabit); `/worker/graph/{id}/completed`: edges + communities (a few MB).
- Remove `/worker/link` calls (worker.py:275/353/394) and `update_edges_for_note` on the primary paths.
  "Rebuild graph" on the primary queues the job (and wakes per the rule).
- Tests: `compute_graph` equals today's `rebuild_graph` on the test corpus; dirty flag; claim order.

### 7. Wake rule

- `WAKE_MIN_JOBS=5` (default 1 = today's behaviour); `wake.tick` wakes when `pending_work()["total"] >=
  WAKE_MIN_JOBS`, counting podcasts, images, index, ingest, insights. A dirty graph alone doesn't wake.
  The WAKE button is unchanged.
- Once awake the PC drains everything, below 5 included, then the graph job, then idles and shuts down.
- Tests: `test_wake.py` thresholds.

### 8. Database churn

- `upsert_note` as `merge_insert` on uuid instead of delete + two adds; `optimize()` once per session (after
  `store_graph`), not per write. Expect far fewer versions than the 253 in half an hour seen today.
- Tests: versions per upsert; data unchanged.

### 9. Docs

`AGENTS.md` ("Two machines"), `deploy/macos/README.md`, `deploy/windows/README.md`, `.env.example`
(`RERANK`, `WAKE_MIN_JOBS`), `HANDOFF.md`.

### 10. The Mac side in containers (OrbStack)

Decided 2026-11-04: containerize the primary only; the worker stays native (Docker on Windows starts at
sign-in, can't see Windows sessions or run `shutdown /s` for the idle shutdown, and reads `D:`/OneDrive media
and multi-GB models slowly through WSL2).

- Done early (2026-11-04), so the Mac's restarts need no sudo; measured in the container: search
  0.4-0.6 s, API 2.3 GB with `MODEL_DTYPE=float16`.
- Planned after step 6 (no heavy work left on the Mac). A Linux container has no MPS: the query model runs on the CPU,
  ~250 ms a query (measured; try float32, bf16 on CPU is slow).
- Two services next to Home Assistant in the existing OrbStack compose (`~/MacContainers`): `zettlecast-api`
  (Python 3.12 slim, `uv pip install .` without `[gpu]`, CPU torch wheels) and `zettlecast-web` (Node 24,
  `npm run build`, `next start -p 3100`).
- **Host networking** for the API: Wake-on-LAN broadcasts to 192.168.8.255, and the PC reaches it at
  192.168.8.140:8000 (as `homeassistant` does). The web UI can be host networked too (port 3100; 3000 is
  zwave-js-ui).
- Mounts: `~/_BRAIN_STORAGE` (read-write), the Hugging Face cache (read-only once the model is in it), `.env`
  and `frontend/.env.local` as env files. Same uid as the Mac user so files stay readable outside.
- `restart: unless-stopped` replaces the launchd daemons; `deploy/macos/install.sh uninstall` removes them
  once the containers are proven. Updates: `git pull && docker compose up -d --build`, no sudo.
- Check before switching: search time, a batch of worker results written through the mount (LanceDB on
  OrbStack's file sharing), memory of the OrbStack VM with HA + Zettlecast, WAKE from the process page.
- Tests: `pytest` inside the API image; the "Checking the whole thing" list below against the containers.

## Deploying

- Primary and worker must run the same code from step 3 on. The PC needs the branch: either push it to
  GitHub and `git pull` on the PC (needs your OK to push), or copy files over `ssh zpc` as now.
- Order per release: pause the worker (process page Stop), update + restart the Mac (`install.sh`), update the
  PC (`git pull`, restart the scheduled task), resume.
- The current Reddit backlog: let it finish, or pause until step 1 is in (it removes the churn either way).

## Checking the whole thing

1. PC off: browse, open the graph, Today; search times < 1 s (first one too); Mac CPU idle between requests.
2. Add 4 items (a page, a PDF, a Reddit sync, a card note): they show as waiting, the PC stays off.
3. A 5th item wakes the PC; it drains the queues, runs one graph job, the items become searchable, the
   graph shows their links, and the PC shuts down after `WORKER_IDLE_SHUTDOWN_MINUTES`.
4. During the session the Mac's API stays well under one core except while writing results; no embedder
   load beyond the query model; `pytest tests` passes on both machines.

## Risks

- **Queued rows** touch everything that reads notes: step 4 audits each reader; a missed one shows a
  zero-vector note in the graph or Today. Tests cover the known readers.
- **Version skew** between the machines: deploy both together; the `kinds` list keeps an old worker safe.
- **Vectors from CUDA, queries on MPS**: same model; float16 vs float32 differences are tiny (the current
  database was built on the PC and searched on the Mac today).
- **Graph input size** grows with chunks (Arrow keeps it a straight copy; ~4 KB per chunk).
