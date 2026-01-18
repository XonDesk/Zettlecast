"""
Zettlecast CLI
Command-line interface for common operations.
"""

import sys
from pathlib import Path

import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Zettlecast - Digital Zettelkasten Middleware"""
    pass


@cli.command()
@click.option("--port", "-p", default=8000, help="API server port")
@click.option("--ui-port", default=8501, help="Streamlit UI port")
@click.option("--no-ui", is_flag=True, help="Start API only, no UI")
def serve(port: int, ui_port: int, no_ui: bool):
    """Start the Zettlecast server."""
    import subprocess
    import signal
    import os
    
    from .config import settings
    
    click.echo("🧠 Starting Zettlecast...")
    settings.ensure_directories()
    
    processes = []
    
    # Start API
    click.echo(f"Starting API server on port {port}...")
    api_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "zettlecast.main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
    ])
    processes.append(api_proc)
    
    # Start UI (optional)
    if not no_ui:
        click.echo(f"Starting UI on port {ui_port}...")
        ui_proc = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            str(Path(__file__).parent / "ui" / "app.py"),
            "--server.port", str(ui_port),
            "--server.headless", "true",
        ])
        processes.append(ui_proc)
    
    click.echo(f"\n✅ Zettlecast running!")
    click.echo(f"   API: http://localhost:{port}")
    if not no_ui:
        click.echo(f"   UI:  http://localhost:{ui_port}")
    click.echo(f"\n🔑 API Token: {settings.api_token[:16]}...")
    click.echo("\nPress Ctrl+C to stop")
    
    def shutdown(signum, frame):
        click.echo("\nShutting down...")
        for p in processes:
            p.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Wait for processes
    for p in processes:
        p.wait()


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def ingest(path: str):
    """Ingest a file or directory."""
    from .parser import parse_file
    from .db import db
    from .chunker import create_chunks
    from .identity import compute_content_hash
    from .models import NoteModel
    
    path = Path(path)
    db.connect()
    
    if path.is_file():
        files = [path]
    else:
        # Find all supported files
        files = list(path.glob("**/*.pdf")) + \
                list(path.glob("**/*.md")) + \
                list(path.glob("**/*.mp3")) + \
                list(path.glob("**/*.m4a"))
    
    click.echo(f"Found {len(files)} files to process")
    
    for file_path in files:
        click.echo(f"Processing: {file_path.name}...", nl=False)
        result = parse_file(file_path)
        
        if result.status == "success":
            click.echo(f" ✅ {result.uuid[:8]}")
        else:
            click.echo(f" ❌ {result.error_message}")


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, help="Number of results")
def search(query: str, top_k: int):
    """Search notes."""
    from .search import search as do_search
    from .db import db
    
    db.connect()
    
    results = do_search(query=query, top_k=top_k)
    
    if not results:
        click.echo("No results found.")
        return
    
    click.echo(f"\n📚 Found {len(results)} results for '{query}':\n")
    
    for i, r in enumerate(results, 1):
        click.echo(f"{i}. [{r.source_type}] {r.title}")
        click.echo(f"   Score: {r.score:.3f}")
        click.echo(f"   {r.snippet[:100]}...")
        click.echo()


@cli.command()
@click.argument("url")
def add(url: str):
    """Quick-add a URL."""
    from .parser import parse_url
    from .db import db
    
    db.connect()
    
    click.echo(f"Ingesting: {url}...", nl=False)
    result = parse_url(url)
    
    if result.status == "success":
        click.echo(f" ✅")
        click.echo(f"   Title: {result.title}")
        click.echo(f"   UUID: {result.uuid}")
    else:
        click.echo(f" ❌ {result.error_message}")


@cli.command()
def token():
    """Print the current API token."""
    from .config import settings
    click.echo(f"API Token: {settings.api_token}")
    click.echo(f"\nBookmarklet (copy this to a bookmark):")
    click.echo(f"javascript:(function(){{fetch('http://localhost:{settings.api_port}/ingest?token={settings.api_token}&url='+encodeURIComponent(location.href))}})();")


@cli.command()
def stats():
    """Show database statistics."""
    from .db import db
    
    db.connect()
    
    try:
        notes = db.notes.search().limit(10000).to_list()
        rejected = db.rejected_edges.search().limit(10000).to_list()
        
        click.echo("📊 Zettlecast Statistics")
        click.echo("=" * 40)
        click.echo(f"Total notes: {len(notes)}")
        click.echo(f"Rejected edges: {len(rejected)}")
        
        # Count by type
        by_type = {}
        for n in notes:
            t = n.get("source_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        
        click.echo("\nBy source type:")
        for t, count in sorted(by_type.items()):
            click.echo(f"  {t}: {count}")
    except Exception as e:
        click.echo(f"Error: {e}")


# =============================================================================
# Podcast Transcription Commands
# =============================================================================

@cli.group()
def podcast():
    """Podcast transcription pipeline commands."""
    pass


@podcast.command("add")
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", "-n", help="Podcast/show name")
@click.option("--recursive/--no-recursive", default=True, help="Scan subdirectories")
def podcast_add(path: str, name: str, recursive: bool):
    """Add audio files to transcription queue."""
    from .podcast.queue import TranscriptionQueue, DuplicateEpisodeError
    
    path = Path(path)
    queue = TranscriptionQueue()
    
    if path.is_file():
        try:
            job_id = queue.add(path, podcast_name=name)
            click.echo(f"✅ Added: {path.name} (ID: {job_id[:8]})")
        except DuplicateEpisodeError:
            click.echo(f"⏭️  Skipped (already processed): {path.name}")
    else:
        job_ids = queue.add_directory(path, podcast_name=name, recursive=recursive)
        click.echo(f"✅ Added {len(job_ids)} episodes to queue")
    
    # Show queue status
    status = queue.get_status_summary()
    click.echo(f"\n📋 Queue: {status['by_status']['pending']} pending, ETA: {status['estimated_remaining']}")


@podcast.command("import")
@click.argument("url")
@click.option("--limit", "-l", default=5, help="Max episodes to download")
def podcast_import(url: str, limit: int):
    """Import episodes from RSS feed URL."""
    from .podcast.queue import TranscriptionQueue
    
    click.echo(f"📡 Fetching feed: {url}...")
    queue = TranscriptionQueue()
    
    try:
        job_ids = queue.add_from_feed(url, limit=limit)
        
        if job_ids:
            click.echo(f"✅ Added {len(job_ids)} episodes to queue")
            
            # Show queue status
            status = queue.get_status_summary()
            click.echo(f"\n📋 Queue: {status['by_status']['pending']} pending, ETA: {status['estimated_remaining']}")
            click.echo("Run 'zettlecast podcast run' to process them")
        else:
            click.echo("⚠️  No new episodes added (duplicates or empty feed)")
            
    except Exception as e:
        click.echo(f"❌ Failed to import feed: {e}")


@podcast.command("run")
@click.option("--limit", "-l", type=int, help="Max episodes to process")
@click.option("--no-enhance", is_flag=True, help="Skip LLM enhancement")
@click.option("--backend", "-b", type=str, help="Force specific backend (parakeet-mlx, mlx-whisper, whisper, nemo)")
@click.option("--no-sync", is_flag=True, help="Skip auto-sync with storage (just process existing queue)")
def podcast_run(limit: int, no_enhance: bool, backend: str, no_sync: bool):
    """Process pending episodes in queue.
    
    Automatically syncs queue with storage: scans downloaded podcasts,
    compares against transcripts, and adds missing items to queue.
    Use --no-sync to skip this and just process existing queue items.
    """
    from .podcast.queue import TranscriptionQueue
    from .podcast.enhancer import TranscriptEnhancer
    from .podcast.formatter import save_result
    from .podcast.transcriber_factory import TranscriberFactory
    from .config import settings
    import time
    
    queue = TranscriptionQueue()
    
    # Auto-sync queue with storage (unless disabled)
    if not no_sync:
        click.echo("🔄 Syncing queue with storage...")
        sync_stats = queue.sync_with_storage()
        
        if sync_stats["reset_stuck"] > 0:
            click.echo(f"   Reset {sync_stats['reset_stuck']} stuck processing items")
        if sync_stats["added_to_queue"] > 0:
            click.echo(f"   Added {sync_stats['added_to_queue']} new episodes from downloads")
        
        click.echo(f"   📁 {sync_stats['podcasts_found']} podcasts, {sync_stats['transcripts_found']} transcripts")
    
    pending = queue.get_pending_count()
    
    if pending == 0:
        click.echo("✅ All podcasts are transcribed!")
        return
    
    # Create transcriber using factory (respects ASR_BACKEND setting)
    transcriber = TranscriberFactory.create(backend=backend)
    caps = transcriber.get_capabilities()
    
    click.echo("=" * 60)
    click.echo("🎙️  Zettlecast Podcast Transcription")
    click.echo("=" * 60)
    click.echo(f"   Backend:     {caps.transcriber_name}")
    click.echo(f"   Diarization: {caps.diarizer_name or 'disabled'}")
    click.echo(f"   Episodes:    {min(limit, pending) if limit else pending} of {pending} pending")
    click.echo(f"   Est. time:   {queue.estimate_time_remaining()}")
    click.echo("=" * 60)
    click.echo("")
    
    # Initialize enhancer
    enhancer = TranscriptEnhancer() if not no_enhance else None
    
    processed = 0
    total_time = 0
    start_time = time.time()
    
    while True:
        if limit and processed >= limit:
            break
        
        item = queue.get_next_pending()
        if not item:
            break
        
        episode = item.episode
        episode_num = processed + 1
        total_to_process = min(limit, pending) if limit else pending
        
        click.echo(f"[{episode_num}/{total_to_process}] {episode.episode_title[:50]}...")
        click.echo(f"        ↳ Downloading... ", nl=False)
        
        queue.mark_started(episode.id)
        
        try:
            # Transcribe
            click.echo("Transcribing... ", nl=False)
            result = transcriber.transcribe(
                Path(episode.audio_path),
                episode=episode,
            )
            
            # Enhance (optional)
            enhanced = None
            if enhancer:
                click.echo("Enhancing... ", nl=False)
                enhanced = enhancer.enhance(result.full_text)
                result.keywords = enhanced.get("keywords", [])
                result.sections = enhanced.get("sections", [])
            
            # Save
            output_path = save_result(result, episode, enhanced)
            queue.mark_completed(episode.id, result, output_path)
            
            total_time += result.processing_time_seconds
            
            click.echo(f"✅ Done!")
            click.echo(f"        ↳ Duration: {result.duration_seconds/60:.1f}min | "
                      f"Time: {result.processing_time_seconds:.0f}s | "
                      f"Speakers: {result.speakers_detected}")
            processed += 1
            
        except Exception as e:
            queue.mark_failed(episode.id, str(e), max_retries=settings.podcast_max_retries)
            click.echo(f"❌ Failed: {str(e)[:60]}")
    
    # Summary
    elapsed = time.time() - start_time
    click.echo("")
    click.echo("=" * 60)
    click.echo(f"✅ Completed: {processed} episodes in {elapsed/60:.1f} minutes")
    if processed > 0:
        click.echo(f"   Avg time per episode: {total_time/processed:.0f}s")
    click.echo("=" * 60)
    
    # Show remaining
    status = queue.get_status_summary()
    if status["by_status"]["pending"] > 0:
        click.echo(f"📋 Remaining: {status['by_status']['pending']} pending")
    if status["by_status"]["review"] > 0:
        click.echo(f"⚠️  {status['by_status']['review']} episodes need review (use: ./zc podcast retry)")


@podcast.command("status")
def podcast_status():
    """Show queue status and time estimate."""
    from .podcast.queue import TranscriptionQueue
    
    queue = TranscriptionQueue()
    status = queue.get_status_summary()
    
    click.echo("📊 Podcast Queue Status")
    click.echo("=" * 40)
    click.echo(f"Total items: {status['total']}")
    click.echo(f"  Pending:    {status['by_status']['pending']}")
    click.echo(f"  Processing: {status['by_status']['processing']}")
    click.echo(f"  Completed:  {status['by_status']['completed']}")
    click.echo(f"  Review:     {status['by_status']['review']}")
    click.echo(f"\nEstimated time remaining: {status['estimated_remaining']}")
    click.echo(f"Avg processing time: {status['avg_processing_time']}")


@podcast.command("retry")
def podcast_retry():
    """Retry failed episodes marked for review."""
    from .podcast.queue import TranscriptionQueue
    
    queue = TranscriptionQueue()
    count = queue.retry_failed()
    
    if count > 0:
        click.echo(f"✅ Reset {count} failed episodes for retry")
        click.echo("Run 'zettlecast podcast run' to process them")
    else:
        click.echo("No failed episodes to retry")


def main():
    cli()


if __name__ == "__main__":
    main()

