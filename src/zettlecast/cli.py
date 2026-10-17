"""
Zettlecast CLI
"""

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


@cli.command()
@click.option("--port", "-p", default=None, type=int, help="API port (default: API_PORT)")
@click.option("--no-frontend", is_flag=True, help="Only start the API")
def serve(port: int, no_frontend: bool):
    """Start the API server and the web UI."""
    from .config import settings

    settings.ensure_directories()
    port = port or settings.api_port

    procs = [subprocess.Popen([
        sys.executable, "-m", "uvicorn", "zettlecast.main:app",
        "--host", settings.api_host, "--port", str(port),
    ])]
    frontend = PROJECT_ROOT / "frontend"
    if not no_frontend and (frontend / "node_modules").exists():
        procs.append(subprocess.Popen("npm run dev", cwd=frontend, shell=True))

    click.echo(f"API:      http://localhost:{port}  (docs: /docs)")
    if len(procs) > 1:
        click.echo("Web UI:   http://localhost:3000")
    click.echo("Press Ctrl+C to stop")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        from . import llm
        llm.unload()


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
    """Recompute all graph edges."""
    from .linker import rebuild_graph

    click.echo(f"Graph rebuilt: {rebuild_graph()} edges")


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
@click.option("--no-sync", is_flag=True, help="Don't scan storage/podcasts for new audio first")
def podcast_run(limit: int, no_enhance: bool, no_sync: bool):
    """Transcribe, diarize, enhance and index pending episodes."""
    from .pipeline import run_podcast_queue
    from .podcast.queue import TranscriptionQueue

    queue = TranscriptionQueue()
    if not no_sync:
        queue.sync_with_storage()
    pending = queue.get_pending_count()
    if pending == 0:
        click.echo("Nothing pending.")
        return
    total = min(limit, pending) if limit else pending
    click.echo(f"Processing {total} of {pending} pending episode(s)\n")

    count = {"n": 0}

    def start(_id, title):
        count["n"] += 1
        click.echo(f"[{count['n']}/{total}] {title}")

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

    summary = run_podcast_queue(
        limit=limit,
        enhance=not no_enhance,
        sync=False,
        on_episode_start=start,
        on_episode_done=done,
        progress_for=progress_for,
    )
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
def podcast_reprocess(match: str, all_: bool):
    """Queue completed episodes (title containing MATCH, or --all) to be processed again."""
    from .podcast.queue import TranscriptionQueue

    if not match and not all_:
        raise click.UsageError("Give a title to match or --all")
    count = TranscriptionQueue().reprocess(None if all_ else match)
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
        result = parse_image(Path(item.image.image_path))
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
    cli()


if __name__ == "__main__":
    main()
