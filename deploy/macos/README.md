# The Mac mini as the primary

The always-on Mac runs the API, the web UI and the database, and holds the notes and the job
queues (`ROLE=primary`). It does no heavy work: no LLM, no embedding of notes, no parsing of pages and
files, no graph rebuilds and no reranking. All of that waits for the PC; what you add is listed at once
("waiting for the desktop") and becomes searchable after the PC's next session. Podcasts, images and LLM tagging go to the
Windows PC (the **worker**, see `deploy/windows/README.md`), which the Mac wakes with Wake-on-LAN
when jobs are waiting.

Search runs on the Mac: vector search with the query model only (no reranker). In the container it
runs on the CPU in float16, ~0.4-0.6 s a search; kept loaded (`MODEL_IDLE_MINUTES=0`).

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
- `insights.json` (Today's cards and your verdicts), `notes/` (notes written from a card)
- each only if it exists; `thumbs/` isn't needed (thumbnails are remade on first view)

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

## 4. Run it: containers (this Mac) or launchd

**In OrbStack containers (how this Mac runs it since 2026-11-04).** Keep the checkout outside
`~/Documents` (launchd-started processes can't read it): `~/Zettlecast`. The two services are in
`~/MacContainers/docker-compose.macos.yaml` (`zettlecast-api`, `zettlecast-web`), started at sign-in
with the rest of that stack.

```bash
./deploy/macos/install-wake-helper.sh      # once: native Wake-on-LAN sender (LaunchAgent, no sudo)
cd ~/MacContainers && docker compose -f docker-compose.macos.yaml up -d --build zettlecast-api zettlecast-web
docker logs -f zettlecast-api
```

- Updates: `git pull` in `~/Zettlecast`, then the same `up -d --build`. No sudo.
- The API runs as your user with `~/_BRAIN_STORAGE` mounted at the same path (notes refer to their
  files by absolute path), the Hugging Face cache at `/hf`, and `.env` mounted (Settings writes it).
- No Apple GPU in a Linux container: the query model runs on the CPU in float16 (`MODEL_DTYPE`):
  searches take ~0.4-0.6 s, the API ~2.3 GB. bfloat16 is ~80x slower there; float32 needs ~1 GB more.
- A container's broadcasts don't reach the LAN: with `WAKE_SENDER=helper` (set in the image) the API
  writes wake requests to `<storage>/wake_requests/`, and `com.zettlecast.wake` sends them
  (`~/Library/Logs/Zettlecast/wake.log`).

**With launchd instead** (native: Apple GPU search ~0.1 s, but starting and restarting needs sudo):


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

- whether the PC is on, and what is waiting for it (and whether the graph needs a rebuild)
- the last wake-up: automatic once 5 items wait (`WAKE_MIN_JOBS`); once on, the PC does everything
  waiting, then rebuilds the graph once, then shuts down when idle
- **WAKE**, **STAY ON** (no idle shutdown) and **RESUME** (after a Stop)

Things to know while the PC is off:

- Notes, search, the graph and Today all work, with what the PC has processed so far.
- New pages, PDFs, fetched links and Reddit saves are listed at once with **waiting for the desktop**:
  not searchable or in the graph until the PC has read, tagged and indexed them. A note that changed
  (a save's comments came in) stays searchable with its earlier version meanwhile.
- Uploaded audio and images wait in the queue.
- A page the PC couldn't read after three tries says **couldn't be added** and why.
- Audio playback and screenshot triage say the desktop is off and offer to wake it.
