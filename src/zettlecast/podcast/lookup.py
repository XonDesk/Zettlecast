"""
Online name lookup: real spellings for names the speech recogniser and the LLM got wrong.

Searches Wikipedia for what the passage says about a person ("Norwegian triathlete").
The result pages and the pages they link to are real names in that field:

- any of them confirms a name exactly as heard or as the LLM proposed it
- a result page itself may also be proposed as a new spelling, but only for a
  multi-word name, with the same number of words and a close spelling
  ("Christian Blunfeld" -> "Kristian Blummenfelt"); the big linked-page pool and
  plain spelling search offered a near-miss for almost anything
  ("Stephen Seiler" -> "Stephen Silver", "Chamonix" -> "Chimney"), so they never propose

The verifier still checks every candidate against the audio. Results are cached
on disk, and the lookup switches itself off for the run after a network error.
"""

import difflib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import httpx

from ..wikipedia import JsonCache, new_client, search_pages

logger = logging.getLogger(__name__)

PROPOSE_CUTOFF = 0.7  # "Christian Blunfeld" ~ "Kristian Blummenfelt" is 0.79
MAX_CANDIDATES = 2
_NAME = re.compile(r"[A-ZÀ-Þ][\w'’.-]+(?: (?:[A-ZÀ-Þ][\w'’.-]+|van|von|de|der|den|du|la|le))*")
_DISAMBIGUATION = re.compile(r" \([^)]*\)$")


def _norm(text: str) -> str:
    return re.sub(r"[^\w ]", "", text.lower()).strip()


def name_titles(titles) -> Set[str]:
    """Article titles that look like names: "Jim Walmsley (runner)" -> "Jim Walmsley"."""
    out = set()
    for title in titles:
        title = _DISAMBIGUATION.sub("", title)
        if _NAME.fullmatch(title) and len(title.split()) <= 4:
            out.add(title)
    return out


def best_matches(heard: str, names: Set[str], limit: int = MAX_CANDIDATES) -> List[str]:
    """Multi-word names with the heard name's word count and a close spelling."""
    heard_n = _norm(heard)
    n_words = len(heard_n.split())
    if n_words < 2:
        return []
    scored = {}
    for name in names:
        name_n = _norm(name)
        if len(name_n.split()) != n_words or name_n == heard_n:
            continue
        ratio = difflib.SequenceMatcher(None, heard_n, name_n).ratio()
        if ratio >= PROPOSE_CUTOFF:
            scored[name] = ratio
    return [n for n, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]


class NameLookup:
    """Wikipedia topic search with an on-disk cache of query -> (result names, linked names)."""

    def __init__(self, cache_path: Optional[Path] = None, client: Optional[httpx.Client] = None):
        self.cache = JsonCache(cache_path)
        self._client = client
        self.enabled = True
        self.queries = 0

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = new_client()
        return self._client

    def _fetch(self, about: str) -> Dict[str, List[str]]:
        if about in self.cache:
            return self.cache.get(about)
        if not self.enabled:
            return {"results": [], "linked": []}
        try:
            pages = search_pages(self.client, about, 10, prop="links", pllimit="max", plnamespace=0)
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(f"Name lookup unavailable, continuing without it: {e}")
            self.enabled = False
            return {"results": [], "linked": []}
        self.queries += 1
        results = name_titles(p["title"] for p in pages)
        linked = name_titles(link["title"] for p in pages for link in p.get("links", [])) - results
        entry = {"results": sorted(results), "linked": sorted(linked)}
        self.cache.set(about, entry)
        return entry

    def find(self, heard: str, about: str) -> Tuple[List[str], Set[str]]:
        """
        (proposals, real names) for a heard name: proposals are new spellings from the
        search results; real names (results + linked pages) confirm exact spellings.
        """
        if not about.strip():
            return [], set()
        entry = self._fetch(about)
        results = set(entry["results"])
        return best_matches(heard, results), results | set(entry["linked"])

    def save(self) -> None:
        self.cache.save()
