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

## Known facts and pitfalls (verified, from earlier work)

- Parakeet can't stay loaded across episodes (illegal memory access after a diarization).
- Restarting the LLM server during pass 2 silently degrades the episode in flight.
- Reddit: anonymous ~1 request a minute; quick sync first, then fetch comments (AGENTS.md).
- Bash heredocs mangle `\n`/`\b` in Python snippets; write patch scripts to a file or use Edit.
- `next dev` locks `frontend/app/*`; stop the server before moving route folders.
