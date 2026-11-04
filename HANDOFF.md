# Handoff (updated 2026-11-04)

Read `AGENTS.md` first (architecture, "Two machines" section). This file is the state of the
work and what to do next.

## Where things stand

- `main` has everything: the two-machine split (`ROLE=primary|worker`), the Mac and Windows
  deploy guides, web UI sign-in, Today's cards, the process page, references.
- `pytest tests`: 284 pass. `npm run build` (frontend) clean.
- **Nothing is deployed yet.** The Windows PC still runs everything as `ROLE=standalone`
  (`.env` has no `ROLE` / `WORKER_TOKEN`). Its storage is `D:/code/Zettlecast/_BRAIN_STORAGE`
  (not the `~/_BRAIN_STORAGE` the guides use as an example): ~530 MB `.lancedb`, 29 MB
  `transcripts`, 200 MB `images`, ~250 podcast episodes, Reddit saves, screenshots.

## Next task: the Mac mini as the always-on primary

Follow `deploy/macos/README.md` step by step. Notes for the agent doing it:

1. **Install** (README step 1). No `[gpu]` extra on the Mac; search models run on MPS.
2. **Data** (step 2) has to come from the PC; the user copies it (network share, USB, scp).
   Stop the PC's `serve` first. `migrate-paths --desktop-root "D:/code/Zettlecast/_BRAIN_STORAGE"`,
   `--dry-run` first, with the API stopped.
3. **Secrets** (step 3): generate new `API_TOKEN` and `WORKER_TOKEN` on the Mac
   (`python -c "import secrets; print(secrets.token_urlsafe(32))"`). `WORKER_TOKEN` must also go
   into the PC's `.env`. For `frontend/.env.local` the user can copy `ZC_LOGIN_USER`,
   `ZC_LOGIN_HASH`, `ZC_SESSION_SECRET` from the PC's file (same sign-in); `ZC_API_TOKEN` = the
   Mac's `API_TOKEN`; `ZC_API_URL` = `http://127.0.0.1:8000`. `ZC_LOGIN_HASH` format is in
   `frontend/lib/auth.ts` (pbkdf2_sha256:iterations:salt:hash, base64url) if a new one is wanted.
   `REDDIT_SAVED_FEED` too, if Reddit sync should run on the Mac (it should: it's the primary's job).
4. **Start at boot** (step 4): `install.sh` needs sudo (the user types the password) and macOS
   will ask to let Python and Node accept connections: the user clicks Allow.
5. **Check**: `curl http://127.0.0.1:8000/health`; sign in on `http://<mac>:3000` from a phone on
   the LAN; notes, search, graph, Today work with the PC off; the process page's Desktop panel
   shows the worker as offline.

Then the PC side, `deploy/windows/README.md` (needs someone at the PC: BIOS Wake-on-LAN,
`setup-worker.ps1` as administrator, `.env` with `ROLE=worker`). Put the MAC address it prints into
the Mac's `DESKTOP_MAC_ADDRESS`, then test WAKE from the process page, one podcast and one image job
end to end, and that the PC shuts itself down after `WORKER_IDLE_SHUTDOWN_MINUTES`.

## Not tested on real hardware

The primary/worker split is covered by unit tests (`tests/test_worker.py`, leases, heartbeat,
pausing) but has never run across two machines: Wake-on-LAN, launchd daemons, MPS search,
`desktop:` file forwarding and the idle shutdown are all untried. Expect small fixes; keep them
on a branch and note them here.

## Fixes from the Mac setup (branch `mac-primary-setup`)

- `psutil` was used by `runstate.py`/`system.py`/`llm.py` but not declared (the PC got it through
  `[gpu]`): added to the base dependencies. Without it 6 tests failed on the Mac.
- The Mac (M1, 16 GB, 192.168.8.140) runs Home Assistant with `zwave-js-ui` on host port 3000, so
  the web UI goes on 3100: `install.sh` takes `WEB_PORT` (default 3000).
- Still failing on macOS, Windows-only by nature: `test_llm.py` (stops llama-server with `taskkill`) x2,
  `test_worker.py::test_triage_keep_and_image_result` (a `C:/` path resolved on POSIX; the code runs on the PC).

- `setup-worker.ps1` registered the task with a typed password (`Read-Host`, impossible over SSH) and
  `$env:USERDOMAIN\$env:USERNAME` (WORKGROUP\xondesk over SSH: "No mapping between account names").
  Now S4U logon (no password stored) and `WindowsIdentity.GetCurrent().Name`. Copied to the PC by scp,
  so the PC's checkout shows it modified until this branch is merged and pulled.

- API token no longer in URLs: `proxy.ts` sends `Authorization: Bearer`, the 54 unused `token: str = Query(...)`
  parameters are gone (the middleware checks; `?token=` still accepted, e.g. the bookmarklet), and the access
  log redacts `token=` and skips successful `/system/status` / `/podcast/status` polls. Token rotated.
- `tests/conftest.py` forces `ROLE=standalone`: the primary's `.env` made 11 tests fail.

## Containers (2026-11-04)

The Mac runs Zettlecast in OrbStack (`zettlecast-api`, `zettlecast-web` in
`~/MacContainers/docker-compose.macos.yaml`), from the checkout at `~/Zettlecast`; the launchd
daemons are uninstalled and `~/Documents/Zettlecast` is the old checkout. Deploy a change: push,
`git pull` in `~/Zettlecast` and on the PC, `up -d --build` on the Mac, and on the PC
`C:\Users\XonDe\restart-worker.ps1` (stopping the task alone leaves the worker running).
Wake-on-LAN: the native `com.zettlecast.wake` LaunchAgent (a container's broadcasts don't reach the LAN).
Checked: the PC reaches the API, heartbeat times in local time (TZ), files owned by the Mac user,
a wake request from the container sent by the helper.

## Open issues seen on the first real run (not fixed)

- Reddit image/link posts with only a title (41-108 chars) fail the LLM pass every time ("no usable answer"):
  46 notes, 3 attempts each. Three in a row make the worker think the LLM is down and release its batch, so
  batches end after ~3 notes and each one triggers a full `/worker/link` graph rebuild on the Mac (every
  ~45 s, the M1 API at 100% CPU). Probably: skip the LLM below MIN_TEXT for reddit too, and don't count a
  "no usable answer" as the LLM being down.
- PC: "Sleep after" 1 h on AC and the worker doesn't block sleep (Modern Standby, S0): a long batch with
  nobody at the PC can be cut off. The worker's own idle shutdown makes Windows sleep unnecessary.

## Done 2026-11-04 (from the Mac, over `ssh zpc`)

- SSH: OpenSSH server on the PC (key `~/.ssh/zettlecast_pc`, firewall LAN only, PowerShell shell);
  `ssh zpc` = xondesk@192.168.8.175 (wired; Wi-Fi is .173).
- Data copied to `~/_BRAIN_STORAGE` (file counts match), `migrate-paths` run: 1743 notes, 32228 chunks, 7707 edges.
- Mac `.env` + `frontend/.env.local` filled (sign-in and Reddit feed copied from the PC).
- PC: `.env` backed up to `.env.standalone.bak`, worker settings appended; `setup-worker.ps1` run (no RDP).
- Mac services installed (`WEB_PORT=3100`), worker task started: PC online, tagging 931 Reddit saves.
- Left: BIOS WoL, end-to-end tests.
- Router: DHCP reservation for the PC (B4:E9:B8:F8:12:7C -> .175).

## Known facts and pitfalls (verified, from earlier work)

- Parakeet can't stay loaded across episodes (illegal memory access after a diarization).
- Restarting the LLM server during pass 2 silently degrades the episode in flight.
- Reddit: anonymous ~1 request a minute; quick sync first, then fetch comments (AGENTS.md).
- Bash heredocs mangle `\n`/`\b` in Python snippets; write patch scripts to a file or use Edit.
- `next dev` locks `frontend/app/*`; stop the server before moving route folders.
