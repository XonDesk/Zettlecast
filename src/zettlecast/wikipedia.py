"""
Wikipedia search, shared by the podcast name lookup (podcast/lookup.py) and the References
card (references.py): one request shape, one user agent, one on-disk cache format.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .podcast.rss import USER_AGENT

logger = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"


def new_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=15)


def search_pages(client: httpx.Client, query: str, limit: int, **props: Any) -> List[dict]:
    """Pages Wikipedia's search finds for query, best match first; props are the page
    properties to return (prop=..., inprop=...). Raises httpx.HTTPError, or ValueError for a
    reply that isn't JSON (a maintenance page)."""
    response = client.get(API_URL, params={
        "format": "json", "action": "query", "generator": "search", "gsrsearch": query,
        "gsrlimit": limit, **props,
    })
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {}).values()
    return sorted(pages, key=lambda p: p.get("index", 0))


class JsonCache:
    """query (lower case) -> result, read from disk once and written back by save()."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        self.lock = threading.Lock()
        self.data: Dict[str, Any] = {}
        if path and path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self.data = {}

    @staticmethod
    def key(query: str) -> str:
        return query.strip().lower()

    def get(self, query: str) -> Any:
        with self.lock:
            return self.data.get(self.key(query))

    def __contains__(self, query: str) -> bool:
        with self.lock:
            return self.key(query) in self.data

    def set(self, query: str, value: Any) -> None:
        with self.lock:
            self.data[self.key(query)] = value

    def save(self) -> None:
        if not self.path:
            return
        with self.lock:
            text = json.dumps(self.data, indent=1, ensure_ascii=False)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(text, encoding="utf-8")
        except OSError as e:
            logger.warning(f"Could not save {self.path}: {e}")
