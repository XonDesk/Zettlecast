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
        response = httpx.post(f"{API_BASE}{endpoint}", json=data, params=params, timeout=30)
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
        ["📚 Notes", "🔍 Search", "📊 Graph", "⚙️ Settings"],
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
    
    st.info("Graph visualization requires notes in the database. Add some content first!")
    
    # Fetch notes for graph
    notes_data = api_get("/notes", {"limit": 100})
    notes = notes_data.get("notes", [])
    
    if notes:
        # Build simple graph data
        nodes = [{"id": n["uuid"], "label": n["title"][:30]} for n in notes]
        
        st.write(f"**{len(nodes)} nodes** in your knowledge graph")
        
        # For MVP, show as a simple list with connections
        # Full Cytoscape integration comes in Phase 2
        st.json(nodes[:10])
        
        st.caption("Full graph visualization coming in Phase 2!")
    else:
        st.info("No notes to visualize yet.")


elif page == "⚙️ Settings":
    st.header("⚙️ Settings")
    
    settings_data = api_get("/settings")
    
    if settings_data:
        st.subheader("Current Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Embedding Model", value=settings_data.get("embedding_model", ""), disabled=True)
            st.text_input("Reranker Model", value=settings_data.get("reranker_model", ""), disabled=True)
            st.text_input("Whisper Model", value=settings_data.get("whisper_model", ""), disabled=True)
        
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


# --- Note Detail Modal ---
if "selected_note" in st.session_state:
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
