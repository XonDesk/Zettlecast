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


def main():
    cli()


if __name__ == "__main__":
    main()
