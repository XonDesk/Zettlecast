"""
Zettlecast Streamlit UI
Main application with graph view, note viewer, and settings.
"""

import json
import os
import sys
from pathlib import Path

import httpx
import streamlit as st

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from zettlecast.config import settings

# --- Page Config ---
st.set_page_config(
    page_title="Zettlecast",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- API Client ---
API_BASE = f"http://localhost:{settings.api_port}"
API_TOKEN = settings.api_token


def api_get(endpoint: str, params: dict = None) -> dict:
    """Make authenticated GET request to API."""
    params = params or {}
    params["token"] = API_TOKEN
    try:
        response = httpx.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return {}


def api_post(endpoint: str, data: dict = None, params: dict = None) -> dict:
    """Make authenticated POST request to API."""
    params = params or {}
    params["token"] = API_TOKEN
    try:
        # Longer timeout for /ingest which includes LLM enrichment
        timeout = 120 if "/ingest" in endpoint else 30
        response = httpx.post(f"{API_BASE}{endpoint}", json=data, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return {}


# --- Sidebar ---
with st.sidebar:
    st.title("🧠 Zettlecast")
    st.caption("Digital Zettelkasten Middleware")
    
    st.divider()
    
    page = st.radio(
        "Navigation",
        ["📚 Notes", "🎙️ Podcasts", "🔍 Search", "📊 Graph", "⚙️ Settings"],
        label_visibility="collapsed",
    )
    
    st.divider()
    
    # Quick stats
    try:
        notes_data = api_get("/notes", {"limit": 1000})
        note_count = len(notes_data.get("notes", []))
        st.metric("Total Notes", note_count)
    except:
        st.metric("Total Notes", "—")
    
    st.divider()
    
    # Quick ingest
    st.subheader("Quick Add")
    url_input = st.text_input("URL", placeholder="https://example.com/article")
    if st.button("➕ Add URL", use_container_width=True):
        if url_input:
            with st.spinner("Ingesting..."):
                result = api_post("/ingest", params={"url": url_input})
                if result.get("status") == "success":
                    st.success(f"Added: {result.get('title', 'Untitled')}")
                else:
                    st.error(result.get("error", "Failed"))


# --- Pages ---

if page == "📚 Notes":
    st.header("📚 Notes")
    
    # Filters
    col1, col2 = st.columns([3, 1])
    with col1:
        status_filter = st.selectbox(
            "Filter by status",
            ["All", "inbox", "reviewed", "archived"],
        )
    with col2:
        refresh = st.button("🔄 Refresh")
    
    # Fetch notes
    params = {"limit": 100}
    if status_filter != "All":
        params["status"] = status_filter
    
    notes_data = api_get("/notes", params)
    notes = notes_data.get("notes", [])
    
    if not notes:
        st.info("No notes yet. Add some content using the sidebar or CLI!")
    else:
        for note in notes:
            with st.expander(f"**{note['title']}** ({note['source_type']})"):
                st.caption(f"UUID: `{note['uuid']}`")
                st.caption(f"Created: {note['created_at']}")
                
                if st.button("View Details", key=f"view_{note['uuid']}"):
                    st.session_state.selected_note = note['uuid']
                    st.rerun()


elif page == "🎙️ Podcasts":
    st.header("🎙️ Podcast Manager")
    
    # Import
    with st.expander("📥 Import Podcast", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            feed_url = st.text_input("RSS Feed URL", placeholder="https://feeds.simplecast.com/...")
        with col2:
            import_limit = st.number_input("Limit", 1, 50, 5)
            
        if st.button("Fetch Episodes"):
            if feed_url:
                try:
                    from zettlecast.podcast.queue import TranscriptionQueue
                    queue = TranscriptionQueue()
                    
                    with st.spinner("Fetching feed..."):
                        job_ids = queue.add_from_feed(feed_url, limit=import_limit)
                        
                    if job_ids:
                        st.success(f"Added {len(job_ids)} episodes to queue!")
                    else:
                        st.info("No new episodes added (duplicates or empty feed)")
                except Exception as e:
                    st.error(f"Failed to import: {e}")
            else:
                st.warning("Please enter a feed URL")

    st.divider()
    
    # Queue Status
    try:
        from zettlecast.podcast.queue import TranscriptionQueue
        queue = TranscriptionQueue()
        status = queue.get_status_summary()
        
        st.subheader("Queue Status")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pending", status['by_status']['pending'])
        col2.metric("Processing", status['by_status']['processing'])
        col3.metric("Completed", status['by_status']['completed'])
        col4.metric("Review Needed", status['by_status']['review'])
        
        if status['by_status']['pending'] > 0:
            st.info(f"⏱️ Estimated time remaining: {status['estimated_remaining']}")
            
            if st.button("▶️ Process Queue (Run in Background)"):
                # This is a bit hacky for Streamlit - ideally would trigger an async job
                # For now we'll just show a message that CLI is better for long jobs
                st.warning("Please run 'zettlecast podcast run' in your terminal for robust processing.")
        
        st.subheader("Episodes")
        
        # Sort items: processing first, then pending, then failed, then completed
        # But for UI, let's just show recent additions
        items = list(queue.items.values())
        items.sort(key=lambda x: x.added_at, reverse=True)
        
        for item in items[:20]:  # Show last 20
            with st.container():
                cols = st.columns([4, 1, 2])
                cols[0].markdown(f"**{item.episode.episode_title}**")
                cols[0].caption(item.episode.podcast_name)
                
                # Status badge
                color = {
                    "pending": "blue",
                    "processing": "orange",
                    "completed": "green",
                    "failed": "red",
                    "review": "red",
                }.get(item.status, "grey")
                cols[1].markdown(f":{color}[{item.status.upper()}]")
                
                cols[2].caption(str(item.added_at).split(".")[0])
                st.divider()
                
    except ImportError:
        st.error("Podcast module dependencies not installed.")
    except Exception as e:
        st.error(f"Error loading queue: {e}")


elif page == "🔍 Search":
    st.header("🔍 Search")
    
    query = st.text_input("Search query", placeholder="What are you looking for?")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        top_k = st.slider("Results", 1, 20, 5)
    with col2:
        use_rerank = st.checkbox("Use reranking", value=True)
    
    if query and len(query) >= 3:
        with st.spinner("Searching..."):
            results = api_get("/search", {
                "q": query,
                "top_k": top_k,
                "rerank": use_rerank,
            })
        
        if results.get("results"):
            st.success(f"Found {len(results['results'])} results")
            
            for i, r in enumerate(results["results"], 1):
                with st.container():
                    st.markdown(f"### {i}. {r['title']}")
                    st.caption(f"Score: {r['score']:.3f} | Type: {r['source_type']}")
                    st.markdown(f"> {r['snippet']}")
                    st.divider()
        else:
            st.info("No results found. Try a different query.")
    elif query:
        st.warning("Query must be at least 3 characters")


elif page == "📊 Graph":
    st.header("📊 Knowledge Graph")
    
    # Fetch notes for graph
    notes_data = api_get("/notes", {"limit": 1000})
    notes = notes_data.get("notes", [])
    
    if notes:
        # --- Linker Section ---
        st.subheader("🔗 Link Builder")
        
        # Track last link time in session state or a simple file
        from pathlib import Path
        from datetime import datetime
        import json
        
        link_state_file = settings.storage_path / ".linker_state.json"
        
        # Load linker state
        last_link_time = None
        last_link_count = 0
        if link_state_file.exists():
            try:
                state = json.loads(link_state_file.read_text())
                last_link_time = datetime.fromisoformat(state.get("last_run", ""))
                last_link_count = state.get("note_count", 0)
            except:
                pass
        
        # Count notes added since last link
        notes_since_link = 0
        if last_link_time:
            for n in notes:
                note_time = datetime.fromisoformat(n["created_at"].replace("Z", "+00:00").replace("+00:00", ""))
                if note_time > last_link_time:
                    notes_since_link += 1
        else:
            notes_since_link = len(notes)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Notes", len(notes))
        col2.metric("Notes Since Last Link", notes_since_link)
        col3.metric("Last Run", 
                   last_link_time.strftime("%Y-%m-%d %H:%M") if last_link_time else "Never")
        
        # Check for incomplete run
        processed_uuids = set()
        if link_state_file.exists():
            try:
                state = json.loads(link_state_file.read_text())
                processed_uuids = set(state.get("processed_uuids", []))
                if state.get("status") == "in_progress":
                    st.warning(f"⚠️ Previous run was interrupted. {len(processed_uuids)} notes already processed.")
            except:
                pass
        
        # Run linker button
        col1, col2 = st.columns([3, 1])
        with col1:
            run_button = st.button("🔗 Run Linker", use_container_width=True, type="primary")
        with col2:
            resume_mode = st.checkbox("Resume", value=bool(processed_uuids), 
                                     help="Skip already-processed notes")
        
        if run_button:
            with st.spinner(f"Building edges for {len(notes)} notes..."):
                try:
                    from zettlecast.linker import build_edges_for_note
                    
                    total_edges = 0
                    errors = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Filter notes if resuming
                    notes_to_process = notes
                    if resume_mode and processed_uuids:
                        notes_to_process = [n for n in notes if n["uuid"] not in processed_uuids]
                        st.info(f"Resuming: {len(notes_to_process)} remaining of {len(notes)}")
                    else:
                        processed_uuids = set()
                    
                    # Save in-progress state
                    link_state_file.write_text(json.dumps({
                        "status": "in_progress",
                        "started_at": datetime.utcnow().isoformat(),
                        "total_notes": len(notes),
                        "processed_uuids": list(processed_uuids),
                    }))
                    
                    for i, note in enumerate(notes_to_process):
                        try:
                            status_text.text(f"Processing: {note['title'][:40]}...")
                            edges = build_edges_for_note(note["uuid"], top_k=10)
                            total_edges += len(edges)
                            processed_uuids.add(note["uuid"])
                            
                            # Save progress every 10 notes
                            if (i + 1) % 10 == 0:
                                link_state_file.write_text(json.dumps({
                                    "status": "in_progress",
                                    "started_at": datetime.utcnow().isoformat(),
                                    "total_notes": len(notes),
                                    "processed_uuids": list(processed_uuids),
                                    "edge_count": total_edges,
                                }))
                                
                        except Exception as e:
                            errors.append({"uuid": note["uuid"], "title": note["title"], "error": str(e)})
                        
                        progress_bar.progress((i + 1) / len(notes_to_process))
                    
                    # Save final state
                    link_state_file.write_text(json.dumps({
                        "status": "completed",
                        "last_run": datetime.utcnow().isoformat(),
                        "note_count": len(notes),
                        "edge_count": total_edges,
                        "processed_uuids": list(processed_uuids),
                        "errors": errors,
                    }))
                    
                    status_text.empty()
                    
                    if errors:
                        st.warning(f"⚠️ Completed with {len(errors)} errors")
                        with st.expander("View Errors"):
                            for err in errors:
                                st.error(f"**{err['title']}**: {err['error']}")
                    else:
                        st.success(f"✅ Created {total_edges} edges across {len(notes_to_process)} notes!")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Linker failed: {e}")
        
        st.divider()
        
        # --- Graph Visualization ---
        st.subheader("Graph View")
        
        # Build simple graph data
        nodes = [{"id": n["uuid"], "label": n["title"][:30]} for n in notes]
        
        st.write(f"**{len(nodes)} nodes** in your knowledge graph")
        
        # For MVP, show as a simple list with connections
        # Full Cytoscape integration comes in Phase 2
        with st.expander("Node Data (first 10)"):
            st.json(nodes[:10])
        
        st.caption("Full graph visualization coming in Phase 2!")
    else:
        st.info("No notes to visualize yet. Add some content first!")


elif page == "⚙️ Settings":
    st.header("⚙️ Settings")
    
    settings_data = api_get("/settings")
    
    if settings_data:
        # --- Platform Detection ---
        st.subheader("🖥️ Platform Detection")
        
        try:
            from zettlecast.podcast.transcriber_factory import TranscriberFactory
            platform_id = TranscriberFactory.get_platform()
            backends = TranscriberFactory.list_available_backends()
            
            # Platform display
            platform_display = {
                "darwin": "🍎 macOS Apple Silicon",
                "win32+cuda": "🪟 Windows + NVIDIA GPU",
                "linux+cuda": "🐧 Linux + NVIDIA GPU",
                "cpu": "💻 CPU Only",
            }.get(platform_id, platform_id)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Detected Platform:** {platform_display}")
            with col2:
                # Recommended backend
                if platform_id == "darwin":
                    st.success("✅ Recommended: **parakeet-mlx** + **pyannote**")
                elif platform_id in ("win32+cuda", "linux+cuda"):
                    st.success("✅ Recommended: **NeMo** (container)")
                else:
                    st.warning("⚠️ Fallback: **Whisper** (no diarization)")
            
            # Available backends
            with st.expander("Available Backends"):
                for backend in backends:
                    status = "✅" if backend.get("available") else "❌"
                    st.markdown(f"{status} **{backend['name']}** - {backend.get('platform', backend.get('reason', ''))}")
                    if backend.get("diarization"):
                        st.caption(f"   Diarization: {backend['diarization']}")
                        
        except ImportError as e:
            st.warning(f"Platform detection unavailable: {e}")
        
        st.divider()
        
        # --- ASR Configuration ---
        st.subheader("🎙️ Audio Transcription Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            asr_backend = st.selectbox(
                "ASR Backend",
                ["auto", "nemo", "parakeet-mlx", "whisper"],
                index=0,
                help="auto selects the best backend for your platform"
            )
            
            diarization_backend = st.selectbox(
                "Diarization Backend",
                ["auto", "pyannote", "nemo", "none"],
                index=0,
                help="auto uses pyannote on Mac, NeMo on Windows"
            )
        
        with col2:
            whisper_device = st.selectbox(
                "Whisper Device (Fallback)",
                ["auto", "cpu", "cuda", "mps"],
                index=0,
            )
            
            whisper_model = st.text_input(
                "Whisper Model",
                value=settings_data.get("whisper_model", "large-v3-turbo"),
                disabled=True
            )
        
        st.caption("⚠️ Settings are read-only. Edit `.env` file to change configuration.")
        
        # Container status (Windows only)
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", "name=zettlecast-nemo"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                st.success("🐳 NeMo container: **Running**")
            else:
                st.info("🐳 NeMo container: Not running")
                st.caption("Start with: `./run.ps1 -StartContainer`")
        except Exception:
            pass  # Docker not available
        
        st.divider()
        
        # --- Current Configuration (original) ---
        st.subheader("Current Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Embedding Model", value=settings_data.get("embedding_model", ""), disabled=True)
            st.text_input("Reranker Model", value=settings_data.get("reranker_model", ""), disabled=True)
        
        with col2:
            st.text_input("LLM Provider", value=settings_data.get("llm_provider", ""), disabled=True)
            st.number_input("Chunk Size", value=settings_data.get("chunk_size", 512), disabled=True)
            st.checkbox("Context Enrichment", value=settings_data.get("enable_context_enrichment", False), disabled=True)
        
        st.divider()
        st.caption(f"Storage Path: `{settings_data.get('storage_path', '')}`")
        
        st.divider()
        st.subheader("Bookmarklet")
        bookmarklet = f"javascript:(function(){{fetch('{API_BASE}/ingest?token={API_TOKEN}&url='+encodeURIComponent(location.href))}})();"
        st.code(bookmarklet, language="javascript")
        st.caption("Drag this to your bookmarks bar to save pages to Zettlecast")
    else:
        st.error("Could not load settings. Is the API running?")


# --- Note Detail Modal (only on Notes page) ---
if page == "📚 Notes" and "selected_note" in st.session_state:
    uuid = st.session_state.selected_note
    note = api_get(f"/notes/{uuid}")
    
    if note:
        st.divider()
        st.header(f"📝 {note['title']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(note.get("full_text", "")[:2000])
            if len(note.get("full_text", "")) > 2000:
                st.caption("(truncated)")
        
        with col2:
            st.subheader("Metadata")
            st.json(note.get("metadata", {}))
            
            st.subheader("💡 Suggested Links")
            suggestions = note.get("suggestions", [])
            if suggestions:
                for s in suggestions:
                    with st.container():
                        st.markdown(f"**{s['title']}**")
                        st.caption(f"Score: {s['score']:.3f}")
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("✅ Accept", key=f"accept_{s['uuid']}"):
                                api_post(f"/notes/{uuid}/link", {"target_uuid": s["uuid"], "action": "accept"})
                                st.success("Linked!")
                        with c2:
                            if st.button("❌ Reject", key=f"reject_{s['uuid']}"):
                                api_post(f"/notes/{uuid}/link", {"target_uuid": s["uuid"], "action": "reject"})
                                st.info("Rejected")
            else:
                st.info("No suggestions yet")
        
        if st.button("← Back to Notes"):
            del st.session_state.selected_note
            st.rerun()
