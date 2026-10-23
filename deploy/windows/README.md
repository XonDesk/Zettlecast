# The Windows PC as the worker

In the two-machine setup the always-on Mac mini is the **primary**: it runs the API, the web UI
and the database, and holds the notes. This PC is the **worker**. It is normally off. The primary
wakes it with Wake-on-LAN when jobs are waiting (podcasts, images, notes to tag). It then does
the GPU work and sends the results back, and shuts itself down once it has been idle for
`WORKER_IDLE_SHUTDOWN_MINUTES` and nobody is signed in.

The PC keeps:

- the podcast audio and its saved transcriptions (`*.asr.json`)
- the screenshots
- the show glossaries and the Wikipedia lookup cache
- its own copy of the transcript notes

While it is on, the primary serves the audio and screenshot triage from it.

## 1. BIOS / UEFI (by hand)

Names vary by board. Look under Power or Advanced:

- **Wake on LAN / Power On By PCI-E / Resume by LAN**: Enabled
- **ErP / EuP Ready / Deep Sleep**: Disabled (these cut the network card's power when the PC is off)

## 2. Windows (script, as administrator)

```powershell
.\deploy\windows\setup-worker.ps1 -PrimaryIp <the Mac's LAN address> -EnableRdp
```

The script:

- turns Fast Startup off
- enables Wake on Magic Packet on the wired card and prints the **MAC address**
- registers the scheduled task `Zettlecast worker`, which runs at boot as you, signed in or not
- opens the worker's file port to the Mac only
- turns on Remote Desktop (with `-EnableRdp`)

Give the PC a fixed address on the router (a DHCP reservation), so the Mac always finds it.

## 3. `.env` on this PC

```ini
ROLE=worker
PRIMARY_URL=http://<the Mac's LAN address>:8000
WORKER_TOKEN=<the same long random string as on the Mac>
WORKER_PORT=8010
WORKER_IDLE_SHUTDOWN_MINUTES=15
```

The other settings (models, LLM backend, storage folder) stay as they are.

## 4. Check it

```powershell
Start-ScheduledTask "Zettlecast worker"
Get-Content .\worker.log -Wait     # "Worker ... working for http://..."
```

The **Desktop** panel on the web UI's process page should then show **ON · IDLE**. Next, shut the
PC down and press **WAKE** on the panel. The PC should start and report back within a couple of
minutes.

- To work on the PC, connect with Remote Desktop over the VPN.
- While anyone is signed in (connected or disconnected), it does not shut down.
- **STAY ON** on the process page keeps it on even with nobody signed in.

## Running a step by hand

The worker can be stopped with `Stop-ScheduledTask "Zettlecast worker"`. With the task stopped,
`python zc.py worker` runs it in a console instead.

On this PC, `python zc.py podcast run` still works against a local queue, but with `ROLE=worker` the
queue that matters is on the primary. Use the process page instead.
