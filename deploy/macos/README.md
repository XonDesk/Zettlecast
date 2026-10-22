# The Mac mini as the primary

The always-on Mac runs the API, the web UI and the database, and holds the notes and the job
queues (`ROLE=primary`). It never does GPU work. Podcasts, images and LLM tagging go to the
Windows PC (the **worker**, see `deploy/windows/README.md`), which the Mac wakes with Wake-on-LAN
when jobs are waiting.

Search runs on the Mac. The embedder and reranker (about 1.2 GB each in float16) run on Apple's
GPU (MPS) and are unloaded after `MODEL_IDLE_MINUTES` without a search.

From outside the house, connect to the router's WireGuard VPN and open `http://<mac LAN address>:3000`.

## 1. Install

```bash
brew install uv node git
git clone <this repository> ~/Zettlecast && cd ~/Zettlecast
uv venv --python 3.12 && uv pip install -e .        # no [gpu] extra: no NeMo, no CUDA
(cd frontend && npm install)
```

## 2. Move the data over (once)

Stop everything on the PC first. Then copy these from the PC's storage folder
(`~/_BRAIN_STORAGE`, e.g. `C:\Users\you\_BRAIN_STORAGE`) into `~/_BRAIN_STORAGE` on the Mac.

**Copy:**

- `.lancedb/`
- `transcripts/`
- `images/`
- `uploads/`
- `podcast_queue.json`, `image_queue.json`
- `reddit_saved.json`, `reddit_curation.md`

**Leave on the PC:**

- `podcasts/` (audio)
- `glossaries/`, `lookups/`
- `image_triage.json`

Then point the copied notes at their new places. Run this once, with the API stopped, giving the
PC's storage folder exactly as it was there:

```bash
.venv/bin/zettlecast migrate-paths --desktop-root "C:/Users/you/_BRAIN_STORAGE" --dry-run   # counts only
.venv/bin/zettlecast migrate-paths --desktop-root "C:/Users/you/_BRAIN_STORAGE"
```

After that:

- A file that was copied is used from the Mac.
- Everything else becomes `desktop:<path>`, which the Mac fetches from the PC while the PC is on.

## 3. `.env` on the Mac

```ini
ROLE=primary
API_TOKEN=<long random string>
WORKER_TOKEN=<another long random string, the same on the PC>
DESKTOP_MAC_ADDRESS=<printed by setup-worker.ps1 on the PC>
WAKE_BROADCAST=<your LAN's broadcast address, e.g. 192.168.1.255>
WORKER_URL=http://<the PC's LAN address>:8010
# WAKE_WINDOW=01:00-07:00   # only wake the PC for queued jobs at night (Wake button: any time)
USE_MARKER_FALLBACK=false   # no Marker on the Mac: scanned PDFs keep pypdf's text
MODEL_IDLE_MINUTES=5        # memory back for the Mac's other services soon after a search
```

`frontend/.env.local` holds the same sign-in settings as on the PC: `ZC_API_TOKEN` (= `API_TOKEN`),
`ZC_SESSION_SECRET`, `ZC_LOGIN_USER` and `ZC_LOGIN_HASH`.

## 4. Start at boot

```bash
./deploy/macos/install.sh            # asks for your password (sudo)
tail -f ~/Library/Logs/Zettlecast/api.log
```

Both services start at boot, signed in or not, and restart if they stop.

- **Firewall:** macOS asks whether Python and Node may accept incoming connections. Allow both,
  or the PC and your phone can't reach them.
- **Ports:** the API listens on the network (port 8000) so the worker can reach it. Every call
  needs a token. The web UI (port 3000) requires the sign-in.

## Day to day

The **process page** has a **Desktop** panel showing:

- whether the PC is on, and what is waiting for it
- the last wake-up
- **WAKE**, **STAY ON** (no idle shutdown) and **RESUME** (after a Stop)

Things to know while the PC is off:

- Notes, search and the graph all work.
- New pages, PDFs and Reddit saves are stored at once and get their LLM tags later.
- Uploaded audio and images wait in the queue.
- Audio playback and screenshot triage say the desktop is off and offer to wake it.
