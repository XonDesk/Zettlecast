"""
Zettlecast CLI
"""

import json
import subprocess
import sys
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Zettlecast - Digital Zettelkasten with podcast transcription and semantic search."""
    import logging
    import os

    # NeMo's diarizer draws tqdm bars on every run; the CLI prints its own stages
    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    for name in (
        "nemo_logger", "nemo", "lightning", "pytorch_lightning", "sentence_transformers",
        "nv_one_logger", "torch.utils.flop_counter",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def _lan_addresses() -> list:
    import socket

    try:
        return sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
                       if not a[4][0].startswith("127.")})
    except OSError:
        return []


def _frontend_build_stale(frontend: Path) -> bool:
    """True if there's no production build or a source file is newer than it."""
    build = frontend / ".next" / "BUILD_ID"
    if not build.exists():
        return True
    built = build.stat().st_mtime
    sources = [frontend / name for name in ("package.json", "package-lock.json", "next.config.ts", "proxy.ts",
                                            "tsconfig.json", ".env.local")]
    for folder in ("app", "components", "lib", "public"):
        sources += [f for f in (frontend / folder).rglob("*") if f.is_file()]
    return any(f.exists() and f.stat().st_mtime > built for f in sources)


@cli.command()
@click.option("--port", "-p", default=None, type=int, help="API port (default: API_PORT)")
@click.option("--no-frontend", is_flag=True, help="Only start the API")
@click.option("--dev", is_flag=True, help="Web UI in development mode (hot reload; several times slower)")
def serve(port: int, no_frontend: bool, dev: bool):
    """Start the API server and the web UI (a production build, rebuilt when the sources changed)."""
    from .config import settings

    settings.ensure_directories()
    port = port or settings.api_port

    procs = [subprocess.Popen([
        sys.executable, "-m", "uvicorn", "zettlecast.main:app",
        "--host", settings.api_host, "--port", str(port),
    ])]
    frontend = PROJECT_ROOT / "frontend"
    if not no_frontend and (frontend / "node_modules").exists():
        # WEB_LAN: the web UI also answers other devices on the network (sign-in required; the
        # API itself stays on this computer)
        host = " -- -H 0.0.0.0" if settings.web_lan else ""
        if dev:
            procs.append(subprocess.Popen(f"npm run dev{host}", cwd=frontend, shell=True))
        else:
            if _frontend_build_stale(frontend):
                click.echo("Building the web UI (sources changed since the last build)...")
                if subprocess.run("npm run build", cwd=frontend, shell=True).returncode != 0:
                    for proc in procs:
                        proc.terminate()
                    raise click.ClickException("Web UI build failed (see above); `serve --dev` runs it unbuilt")
            procs.append(subprocess.Popen(f"npm run start{host}", cwd=frontend, shell=True))

    click.echo(f"API:      http://localhost:{port}  (docs: /docs)")
    if len(procs) > 1:
        click.echo("Web UI:   http://localhost:3000")
        if settings.web_lan:
            for address in _lan_addresses():
                click.echo(f"          http://{address}:3000  (other devices on the network)")
    click.echo("Press Ctrl+C to stop")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if sys.platform == "win32":
                # npm runs under a shell: terminate() would stop only the shell, leaving node running
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
            else:
                p.terminate()
        from . import llm
        llm.unload(only_own=True)


def _no_gpu_here() -> None:
    """GPU work runs on the worker when this is the primary."""
    from .config import settings

    if settings.is_primary:
        raise click.ClickException("This is the primary (ROLE=primary): the worker does GPU work. "
                                   "Queued jobs wake it (`zettlecast wake` to wake it now).")


@cli.command()
@click.option("--once", is_flag=True, help="One claim and its batch, then exit (testing)")
@click.option("--no-file-server", is_flag=True, help="Don't serve audio / screenshot triage to the primary")
def worker(once: bool, no_file_server: bool):
    """Do the primary's GPU jobs (ROLE=worker): podcasts, images, tagging. Started at boot."""
    import logging

    from . import llm
    from . import worker as worker_module

    logging.getLogger("zettlecast").setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:  # a log with times: it runs for days
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    try:
        worker_module.run(file_server=not no_file_server, once=once)
    except worker_module.PrimaryError as e:
        raise click.ClickException(str(e))
    except KeyboardInterrupt:
        pass
    finally:
        llm.unload(only_own=True)


@cli.command()
def wake():
    """Wake the worker PC now (Wake-on-LAN; run on the primary)."""
    from .wake import wake_now

    result = wake_now("manual")
    click.echo(result["message"])
    if result["status"] in ("failed", "not_configured"):
        sys.exit(1)


@cli.command("migrate-paths")
@click.option("--desktop-root", required=True,
              help="The desktop's storage folder the data was copied from, e.g. C:/Users/me/_BRAIN_STORAGE")
@click.option("--dry-run", is_flag=True, help="Only count what would change")
def migrate_paths(desktop_root: str, dry_run: bool):
    """After copying the storage folder to the primary: point notes at the copies or at the desktop."""
    from .migrate import migrate

    counts = migrate(desktop_root, dry_run=dry_run)
    verb = "Would update" if dry_run else "Updated"
    click.echo(f"{verb} {counts['notes']} note(s), {counts['transcripts']} transcript file(s), "
               f"{counts['queue_items']} queue item(s)")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", "-n", help="Podcast name for audio files")
def ingest(path: str, name: str):
    """Ingest a file or folder (PDF, markdown, text, images; audio is queued for transcription)."""
    from .parser import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, TEXT_EXTENSIONS
    from .pipeline import ingest_path

    path = Path(path)
    supported = AUDIO_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS | {".pdf"}
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in supported
    )
    click.echo(f"Found {len(files)} files")

    queued = 0
    for file_path in files:
        click.echo(f"  {file_path.name} ... ", nl=False)
        result = ingest_path(file_path, podcast_name=name)
        if result.status == "success":
            click.echo(f"ok ({result.uuid[:8]})")
        elif result.status == "queued":
            queued += 1
            click.echo("queued for transcription")
        elif result.status == "duplicate":
            click.echo("already ingested")
        else:
            click.echo(f"FAILED: {result.error_message}")
    if queued:
        click.echo(f"\n{queued} audio file(s) queued. Run: zettlecast podcast run")


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, help="Number of results")
def search(query: str, top_k: int):
    """Semantic search across all notes."""
    from .search import search as do_search

    results = do_search(query=query, top_k=top_k)
    if not results:
        click.echo("No results found.")
        return
    for i, r in enumerate(results, 1):
        at = f" @ {int(r.timestamp_seconds // 60)}:{int(r.timestamp_seconds % 60):02d}" if r.timestamp_seconds is not None else ""
        click.echo(f"{i}. [{r.source_type}] {r.title}{at}  (score {r.score:.3f})")
        click.echo("   " + r.snippet[:300].replace("\n", "\n   "))
        click.echo()


@cli.command()
@click.argument("url")
def add(url: str):
    """Ingest a web page."""
    from .parser import parse_url
    from .pipeline import ingest_note

    result = parse_url(url)
    if result.status == "success":
        ingest_note(result.note)
        click.echo(f"ok: {result.title} ({result.uuid[:8]})")
    else:
        click.echo(f"FAILED: {result.error_message}")


@cli.command()
def token():
    """Print the API token and bookmarklet."""
    from .config import settings

    click.echo(f"API Token: {settings.api_token}")
    click.echo("\nBookmarklet (save as a bookmark URL):")
    click.echo(
        f"javascript:(function(){{fetch('http://localhost:{settings.api_port}/ingest?token="
        f"{settings.api_token}&url='+encodeURIComponent(location.href),{{method:'POST'}})"
        f".then(r=>r.json()).then(j=>alert('Zettlecast: '+j.status))}})();"
    )


@cli.command()
def stats():
    """Show database statistics."""
    from .db import db

    s = db.stats()
    click.echo(f"Notes:  {s['notes']}")
    click.echo(f"Chunks: {s['chunks']}")
    click.echo(f"Edges:  {s['edges']}")
    for t, count in sorted(s["by_source_type"].items()):
        click.echo(f"  {t}: {count}")


@cli.command()
def reindex():
    """Re-read all transcript notes from disk, re-embed them and rebuild the graph."""
    from .pipeline import reindex_all

    click.echo(f"Reindexed {reindex_all()} notes")


@cli.group()
def graph():
    """Knowledge graph commands."""


@graph.command("rebuild")
def graph_rebuild():
    """Recompute all graph edges and communities."""
    from .linker import rebuild_graph

    click.echo(f"Graph rebuilt: {rebuild_graph()} edges")


@graph.command("stats")
def graph_stats():
    """Edges per note, link kinds and the communities with their notes."""
    from collections import Counter

    from .db import db

    titles = {r["uuid"]: r["title"] for r in db.notes.search().select(["uuid", "title"]).limit(1_000_000).to_list()}
    edges = db.get_all_edges()
    degree = Counter(u for e in edges for u in (e["source_uuid"], e["target_uuid"]))
    if degree:
        values = sorted(degree.values())
        click.echo(f"{len(titles)} notes, {len(edges)} edges; links per note min {values[0]}, "
                   f"median {values[len(values) // 2]}, max {values[-1]}; unlinked {len(set(titles) - set(degree))}")
    click.echo("Kinds: " + ", ".join(f"{k} {n}" for k, n in Counter(e["kind"] for e in edges).items()))
    communities = db.get_communities()
    clustered = {u for c in communities for u in c["members"]}
    click.echo(f"{len(communities)} communities; unclustered notes {len(set(titles) - clustered)}")
    for c in sorted(communities, key=lambda c: -len(c["members"])):
        click.echo(f"\n[{c['community']}] {c['label']} ({len(c['members'])} notes)")
        for u in c["members"][:8]:
            click.echo(f"   {titles.get(u, u)[:90]}")
        if len(c["members"]) > 8:
            click.echo(f"   ... {len(c['members']) - 8} more")


@graph.command("explain")
@click.argument("first")
@click.argument("second")
def graph_explain(first: str, second: str):
    """Why two notes are (or aren't) linked; FIRST/SECOND match titles."""
    from .linker import build

    graph = build()

    def find(text: str) -> int:
        hits = [i for i, n in enumerate(graph.notes) if text.lower() in n["title"].lower()]
        if len(hits) != 1:
            raise click.ClickException(f"'{text}' matches {len(hits)} notes; be more specific")
        return hits[0]

    i, j = find(first), find(second)
    for name, row in (("weight", graph.weights), ("semantic (centred cosine)", graph.semantic),
                      ("passages", graph.passage), ("entities", graph.entities.similarity)):
        click.echo(f"{name:26} {row[i, j]:.3f}   (rank among {graph.notes[i]['title'][:30]}'s partners: "
                   f"{int((row[i] > row[i, j]).sum()) + 1})")
    shared = set(graph.entities.per_note[i]) & set(graph.entities.per_note[j])
    click.echo("shared: " + ", ".join(f"{graph.entities.display[k]} ({graph.entities.per_note[i][k]:.2f})"
                                      for k in sorted(shared, key=lambda k: -graph.entities.per_note[i][k])))


# =============================================================================
# Podcast commands
# =============================================================================

@cli.group()
def podcast():
    """Podcast transcription pipeline."""


@podcast.command("add")
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", "-n", help="Podcast/show name")
@click.option("--recursive/--no-recursive", default=True, help="Scan subdirectories")
def podcast_add(path: str, name: str, recursive: bool):
    """Add audio files to the transcription queue."""
    from .podcast.queue import DuplicateEpisodeError, TranscriptionQueue

    path = Path(path)
    queue = TranscriptionQueue()
    if path.is_file():
        try:
            queue.add(path, podcast_name=name)
            click.echo(f"Added: {path.name}")
        except DuplicateEpisodeError:
            click.echo(f"Already queued or processed: {path.name}")
    else:
        job_ids = queue.add_directory(path, podcast_name=name, recursive=recursive)
        click.echo(f"Added {len(job_ids)} episodes")
    click.echo(f"Pending: {queue.get_pending_count()}")


@podcast.command("import")
@click.argument("url")
@click.option("--limit", "-l", default=5, help="Max episodes to download")
def podcast_import(url: str, limit: int):
    """Download the latest episodes of an RSS feed and queue them."""
    from .podcast.queue import TranscriptionQueue

    click.echo(f"Fetching feed: {url}")
    queue = TranscriptionQueue()
    job_ids = queue.add_from_feed(url, limit=limit)
    click.echo(f"Added {len(job_ids)} episodes. Pending: {queue.get_pending_count()}")


@podcast.command("run")
@click.option("--limit", "-l", type=int, help="Max episodes to process")
@click.option("--no-enhance", is_flag=True, help="Skip LLM summary/cleanup")
@click.option("--no-verify", is_flag=True, help="Don't check corrections against the audio")
@click.option("--no-sync", is_flag=True, help="Don't scan storage/podcasts for new audio first")
def podcast_run(limit: int, no_enhance: bool, no_verify: bool, no_sync: bool):
    """Transcribe, diarize, enhance and index pending episodes."""
    from .pipeline import run_podcast_queue
    from .podcast.queue import TranscriptionQueue

    _no_gpu_here()
    queue = TranscriptionQueue()
    if not no_sync:
        queue.sync_with_storage()
    pending = queue.get_pending_count()
    if pending == 0:
        click.echo("Nothing pending.")
        return
    total = min(limit, pending) if limit else pending
    click.echo(f"Processing {total} of {pending} pending episode(s)\n")

    count = {"n": 0, "of": total}

    def on_pass(name, n):
        count.update(n=0, of=n)
        heading = {"transcribe": "Pass 1: transcription", "enhance": "Pass 2: analysis and corrections"}[name]
        click.echo(f"== {heading} ({n} episode(s)) ==")

    def start(_id, title):
        count["n"] += 1
        click.echo(f"[{count['n']}/{count['of']}] {title}")

    def progress_for(_id):
        last = {"stage": None}

        def cb(stage, chunk, total_chunks, device):
            if stage != last["stage"]:
                last["stage"] = stage
                click.echo(f"    {stage}...")
            return True

        return cb

    def done(o):
        if o.status == "completed":
            click.echo(
                f"    done: {o.duration_seconds / 60:.1f} min audio in {o.processing_seconds:.0f}s, "
                f"{o.speakers} speakers -> {o.note_path.name}\n"
            )
        else:
            click.echo(f"    {o.status.upper()}: {o.error or ''}\n")

    from .pipeline import RunActiveError

    try:
        summary = run_podcast_queue(
            limit=limit,
            enhance=not no_enhance,
            verify=not no_verify,
            sync=False,
            on_episode_start=start,
            on_episode_done=done,
            progress_for=progress_for,
            on_pass=on_pass,
            source="cli",
        )
    except RunActiveError as e:
        raise click.ClickException(str(e))
    click.echo(f"Completed {summary.completed}, failed {summary.failed}")


@podcast.command("status")
def podcast_status():
    """Show queue status."""
    from .podcast.queue import TranscriptionQueue

    status = TranscriptionQueue().get_status_summary()
    click.echo(f"Total: {status['total']}")
    for k, v in status["by_status"].items():
        click.echo(f"  {k}: {v}")


@podcast.command("verify")
@click.argument("match", required=False)
@click.option("--spans", "-s", default=8, help="Passages to list for spot-checking")
def podcast_verify(match: str, spans: int):
    """Compare transcripts with the ones published in the feed (title containing MATCH, or all)."""
    from .podcast.queue import TranscriptionQueue
    from .podcast.reference import cache_path_for, compare, load_reference, note_turns

    def mmss(t: float) -> str:
        return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"

    items = [
        i for i in TranscriptionQueue().items.values()
        if i.status == "completed" and i.result_path
        and (not match or match.lower() in (i.episode.episode_title or "").lower())
    ]
    if not items:
        click.echo("No matching processed episodes.")
        return

    totals = {"ref": 0, "diff": 0}
    for item in items:
        ep = item.episode
        click.echo(f"== {ep.episode_title}")
        if not ep.transcript_url:
            click.echo("   no published transcript (import the feed to pick up its transcript link)\n")
            continue
        note = Path(item.result_path)
        if not note.exists():
            click.echo(f"   note missing: {note}\n")
            continue
        try:
            ref = load_reference(ep.transcript_url, cache_path_for(Path(ep.audio_path), ep.transcript_url))
        except Exception as e:
            click.echo(f"   could not load published transcript: {e}\n")
            continue

        r = compare(note_turns(note.read_text(encoding="utf-8")), ref, max_spans=spans)
        totals["ref"] += r.reference_words
        totals["diff"] += r.substitutions + r.missing + r.extra
        click.echo(
            f"   words differ: {r.disagreement:.1%} of {r.reference_words} "
            f"({r.substitutions} different, {r.missing} only in theirs, {r.extra} only in ours)"
        )
        if r.speaker_agreement is not None:
            click.echo(f"   same speaker: {r.speaker_agreement:.1%} of words")
        for theirs, ours in r.name_mismatches:
            click.echo(f"   NAME: publisher says {theirs!r}, we say {ours!r}")
        if r.top_differences:
            click.echo("   most frequent differences (theirs -> ours): " + ", ".join(
                f"{t!r}->{o!r} x{n}" for t, o, n in r.top_differences[:8]
            ))
        if r.spans:
            click.echo("   passages to spot-check:")
            for s in r.spans:
                click.echo(f"     {mmss(s.time)}  theirs: {s.reference}")
                click.echo(f"               ours:   {s.ours}")
        click.echo()
    if totals["ref"]:
        click.echo(f"Overall: words differ on {totals['diff'] / totals['ref']:.1%}")


@podcast.command("reprocess")
@click.argument("match", required=False)
@click.option("--all", "all_", is_flag=True, help="Reprocess every completed episode")
@click.option("--retranscribe", is_flag=True, help="Also redo speech recognition (default: reuse the saved transcription)")
def podcast_reprocess(match: str, all_: bool, retranscribe: bool):
    """
    Queue completed episodes (title containing MATCH, or --all) to be processed again.
    The saved transcription is reused, so only the analysis/corrections pass runs.
    """
    from .podcast.asr_cache import delete_asr
    from .podcast.queue import TranscriptionQueue

    if not match and not all_:
        raise click.UsageError("Give a title to match or --all")
    queue = TranscriptionQueue()
    count = queue.reprocess(None if all_ else match)
    if retranscribe:
        for item in queue.items.values():
            if item.status == "pending":
                delete_asr(Path(item.episode.audio_path))
    click.echo(f"Queued {count} episode(s) again. Run: zettlecast podcast run")


@podcast.command("retry")
def podcast_retry():
    """Reset failed episodes to pending."""
    from .podcast.queue import TranscriptionQueue

    click.echo(f"Reset {TranscriptionQueue().retry_failed()} episodes")


# =============================================================================
# Image commands
# =============================================================================

@cli.group()
def reddit():
    """Reddit saves (from the private saved feed, REDDIT_SAVED_FEED)."""


@reddit.command("sync")
@click.option("--limit", type=int, default=None, help="Import at most N saves (newest first)")
@click.option("--refresh", is_flag=True, help="Also retry saves whose thread couldn't be fetched")
@click.option("--no-articles", is_flag=True, help="Don't read the pages link posts point to")
@click.option("--no-llm", is_flag=True, help="No tags/summary now (GPU busy); `reddit tag` adds them later")
@click.option("--no-threads", is_flag=True, help="No comments now (one request per save); --refresh adds them later")
def reddit_sync(limit: int, refresh: bool, no_articles: bool, no_llm: bool, no_threads: bool):
    """Import saved posts and comments not imported yet."""
    from . import reddit as rd
    from . import runstate

    if runstate.active_run():
        raise click.ClickException("A podcast run is using the GPU: sync after it finishes")
    try:
        counts = rd.sync(limit=limit, refresh=refresh, articles=not no_articles, enrich=not no_llm,
                         threads=not no_threads,
                         on_item=lambda n, total, title: click.echo(f"[{n}/{total}] {title[:90]}"))
    except rd.RedditError as e:
        raise click.ClickException(str(e))
    click.echo(f"{counts['saved']} saves in the feed: {counts['imported']} imported, "
               f"{counts['refreshed']} refreshed, {counts['failed']} failed"
               + (f" ({counts['partial']} without their thread: run with --refresh later)" if counts["partial"] else "")
               + (f"; {counts['unenriched']} without tags (run `reddit tag` when the GPU is free)"
                  if counts["unenriched"] else "")
               + (f"; {counts['tagged']} tagged" if counts["tagged"] else ""))


@reddit.command("tag")
def reddit_tag():
    """Curate comments and add LLM tags + summaries where missing (no Reddit requests)."""
    from . import reddit as rd

    try:
        counts = rd.tag_untagged(on_item=lambda n, total, title: click.echo(f"[{n}/{total}] {title[:90]}"))
    except rd.RedditError as e:
        raise click.ClickException(str(e))
    click.echo(f"{counts['tagged']} tagged, {counts['curated']} curated"
               + (" (stopped: the LLM isn't answering)" if counts["failed"] else ""))


@reddit.command("review")
@click.option("--count", type=int, default=20, help="Saves to curate")
@click.option("--seed", type=int, default=1, help="Another seed picks other saves")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="JSON file to write")
def reddit_review(count: int, seed: int, out: Path):
    """Curate a sample of saves without storing anything, to check the rubric against your judgement."""
    from . import reddit as rd

    reviews = rd.review_sample(count=count, seed=seed)
    out.write_text(json.dumps(reviews, indent=1, ensure_ascii=False), encoding="utf-8")
    kept = sum(v["keep"] for r in reviews for v in r["verdicts"])
    total = sum(len(r["verdicts"]) for r in reviews)
    click.echo(f"{len(reviews)} saves: kept {kept} of {total} comments -> {out}")


@reddit.command("status")
def reddit_status():
    """Imported saves and the last sync."""
    from . import reddit as rd

    s = rd.status()
    click.echo(f"feed: {'set' if s['configured'] else 'not set (REDDIT_SAVED_FEED)'}")
    click.echo(f"imported: {s['imported']} ({s['partial']} without their thread, {s['untagged']} without tags, "
               f"{s['uncurated']} with comments not curated yet)")
    click.echo(f"last sync: {s['last_sync'] or 'never'}")


@cli.group()
def image():
    """Image processing pipeline."""


@image.command("add")
@click.argument("path", type=click.Path(exists=True))
@click.option("--collection", "-c", help="Collection name")
@click.option("--recursive/--no-recursive", default=True, help="Scan subdirectories")
def image_add(path: str, collection: str, recursive: bool):
    """Add images to the processing queue."""
    from .image.queue import DuplicateImageError, ImageQueue

    path = Path(path)
    queue = ImageQueue()
    if path.is_file():
        try:
            queue.add(path, collection_name=collection)
            click.echo(f"Added: {path.name}")
        except DuplicateImageError as e:
            click.echo(str(e))
    else:
        click.echo(f"Added {len(queue.add_directory(path, collection_name=collection, recursive=recursive))} images")


@image.command("run")
@click.option("--limit", "-l", type=int, help="Max images to process")
def image_run(limit: int):
    """Analyse pending images and index them."""
    import time

    _no_gpu_here()
    from .image.image_parser import parse_image
    from .image.queue import ImageQueue
    from .pipeline import ingest_note

    queue = ImageQueue()
    processed = 0
    while not limit or processed < limit:
        item = queue.get_next_pending()
        if not item:
            break
        click.echo(f"{Path(item.image.image_path).name} ... ", nl=False)
        queue.mark_started(item.image.id)
        started = time.time()
        result = parse_image(Path(item.image.image_path), keep_image=item.image.keep_image,
                             focus_region=item.image.focus_region)
        if result.status == "success":
            ingest_note(result.note)
            queue.mark_completed(item.image.id, time.time() - started)
            click.echo(f"ok ({time.time() - started:.1f}s)")
        else:
            queue.mark_failed(item.image.id, result.error_message)
            click.echo(f"FAILED: {result.error_message}")
        processed += 1
    click.echo(f"Processed {processed} images")


@image.command("status")
def image_status():
    """Show queue status."""
    from .image.queue import ImageQueue

    status = ImageQueue().get_status_summary()
    click.echo(f"Total: {status['total']}")
    for k, v in status["by_status"].items():
        click.echo(f"  {k}: {v}")


@image.command("retry")
def image_retry():
    """Reset failed images to pending."""
    from .image.queue import ImageQueue

    click.echo(f"Reset {ImageQueue().retry_failed()} images")


def main():
    # Redirected output defaults to the ANSI code page on Windows, and episode titles
    # can hold any character ("⎮" stopped a whole queue run)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cli()


if __name__ == "__main__":
    main()
