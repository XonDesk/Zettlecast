"""
References: what a note links to or mentions, and fetching one of them as a note of its own.

- links: URLs in the text (markdown links, Reddit's "text (url)", bare URLs) and a Reddit
  post's link; media files, Reddit itself and the note's own source are left out, and links
  to sites that rarely give readable text (video, social, forms, shorteners) are marked
  low_value (listed, but last)
- names: what a screenshot was kept as ("Book: Dune"), and the people and organisations of
  the note; a name is looked up on Wikipedia and the page picked is fetched

A fetched page is an ordinary web note with `referenced_by` (the notes it was fetched from),
which the linker turns into a link edge.
"""

import logging
import re
import threading
from typing import Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from .chunker import UNINDEXED_SECTIONS, _drop_sections
from .config import settings
from .db import db
from .identity import compute_content_hash
from .models import NoteModel, ProcessingResult
from .wikipedia import JsonCache, new_client, search_pages

logger = logging.getLogger(__name__)

LOOKUP_LIMIT = 5
CONTEXT_CHARS = 160

# the most linked sites in the notes that give little or no text: video, images, social, forms, shorteners
LOW_VALUE_DOMAINS = {
    "google.com", "forms.gle", "youtube.com", "youtu.be", "imgur.com", "gfycat.com", "streamable.com",
    "twitter.com", "x.com", "instagram.com", "facebook.com", "tiktok.com", "discord.gg", "discord.com",
    "patreon.com", "soundcloud.com", "spotify.com", "podcasts.apple.com", "linktr.ee",
    "bit.ly", "smarturl.it", "tinyurl.com", "geni.us", "amzn.to",
}
SKIP_DOMAINS = {"reddit.com", "redd.it", "redditmedia.com", "redditstatic.com"}
MEDIA_EXTENSIONS = re.compile(r"\.(jpe?g|png|gif|webp|svg|bmp|mp4|webm|mov|mp3|m4a|wav|ogg|pdf)$", re.I)
TRACKING_PARAMS = re.compile(r"^(utm_\w+|fbclid|gclid|mc_cid|mc_eid|ref_src|igshid)$", re.I)
# what a screenshot kept as one of these is worth looking up
LOOKUP_SAVED_AS = {"media", "product", "place", "reference"}

# the text of a link: [text](url) in markdown, "text (url)" in Reddit notes, or a bare url
# a URL may hold balanced parentheses (en.wikipedia.org/wiki/Dune_(novel)), one level deep
_URL_IN_PARENS = r"https?://(?:[^()\s]|\([^()\s]*\))+"
_MARKDOWN_LINK = re.compile(r"(?<!!)\[([^\]]{1,200})\]\((" + _URL_IN_PARENS + r")\)")
_PAREN_LINK = re.compile(r"([^\s(][^()\n]{0,120}?) \((" + _URL_IN_PARENS + r")\)")
_BARE_CHAR = r"[^\s<>\"'()\]\u200b-\u200f\u2060\ufeff]"
_BARE_URL = re.compile(r"https?://(?:" + _BARE_CHAR + r"|\(" + _BARE_CHAR + r"*\))+")
_INVISIBLE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")  # zero-width characters some show notes put in links
_TRAILING = ".,;:!?'\""

def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().split(":")[0].removeprefix("www.").removeprefix("m.")


def _in(domain: str, domains: set) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in domains)


def clean_url(url: str) -> str:
    """The URL as linked, without zero-width characters and sentence punctuation after it."""
    return _INVISIBLE.sub("", url).strip().rstrip(_TRAILING)


def normalise_url(url: str) -> str:
    """Without tracking parameters, fragment and trailing slash: a key to compare links by,
    not an address to fetch (a site may need the slash, a signed link its exact query)."""
    parts = urlparse(clean_url(url))
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if not TRACKING_PARAMS.match(k)])
    path = parts.path.rstrip("/") if parts.path != "/" else ""
    return urlunparse((parts.scheme.lower(), parts.netloc.lower(), path, parts.params, query, ""))


def _context(text: str, start: int, end: int) -> str:
    """The line around a link, shortened to about CONTEXT_CHARS."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start:line_end if line_end >= 0 else len(text)].strip()
    if len(line) > CONTEXT_CHARS:
        middle = start - line_start
        lo = max(0, middle - CONTEXT_CHARS // 2)
        line = ("…" if lo else "") + line[lo:lo + CONTEXT_CHARS].strip() + "…"
    return line


def _label(url: str) -> str:
    parts = urlparse(url)
    path = parts.path.rstrip("/")
    return _domain(url) + (path if len(path) <= 60 else path[:57] + "…")


def extract_links(note: NoteModel) -> List[dict]:
    """The URLs a note links to, first mention first (labelled ones keep their text)."""
    text = _drop_sections(note.full_text or "", tuple(s for s in UNINDEXED_SECTIONS if s != "Thread comments"))
    own = {normalise_url(u) for u in (note.source_path, note.metadata.source_url) if u and u.startswith("http")}
    found: Dict[str, dict] = {}

    def add(url: str, label: str, start: int, end: int) -> None:
        url = clean_url(url)
        key = normalise_url(url)
        domain = _domain(url)
        if (key in own or not domain or _in(domain, SKIP_DOMAINS)
                or MEDIA_EXTENSIONS.search(urlparse(url).path)):
            return
        entry = found.get(key)
        if entry is None:
            found[key] = entry = {"kind": "url", "url": url, "label": "", "domain": domain,
                                  "context": _context(text, start, end), "position": start,
                                  "low_value": _in(domain, LOW_VALUE_DOMAINS)}
        if label and not entry["label"]:
            label = label.strip().lstrip("-*> ").strip()
            if len(label) > 80:  # "text (url)" in the middle of a sentence: the words just before the link
                label = "…" + label[-80:].split(" ", 1)[-1]
            entry["label"] = label

    covered = []
    for pattern in (_MARKDOWN_LINK, _PAREN_LINK):
        for m in pattern.finditer(text):
            add(m.group(2), m.group(1), m.start(), m.end())
            covered.append((m.start(2), m.end(2)))
    for m in _BARE_URL.finditer(text):
        if not any(a <= m.start() < b for a, b in covered):
            add(m.group(0), "", m.start(), m.end())

    link_url = ((note.metadata.custom or {}).get("reddit") or {}).get("link_url")
    if link_url:
        add(link_url, "Linked post", 0, 0)

    links = sorted(found.values(), key=lambda e: (e["low_value"], e["position"]))
    for entry in links:
        entry["label"] = entry["label"] or _label(entry["url"])
        del entry["position"]
    return links


def extract_names(note: NoteModel) -> List[dict]:
    """What a screenshot was kept as, then the people and organisations of a note."""
    custom = note.metadata.custom or {}
    names: List[dict] = []
    seen = set()

    def add(label: str, query: str, what: str) -> None:
        key = label.strip().lower()
        if key and key not in seen:
            seen.add(key)
            names.append({"kind": "name", "label": label.strip(), "query": query.strip(), "what": what})

    item = custom.get("item") or {}
    if item.get("name") and custom.get("saved_as") in LOOKUP_SAVED_AS:
        query = item["name"] + (f" {item['by']}" if item.get("by") else "")
        add(item["name"], query, item.get("type") or custom["saved_as"])
    for person in custom.get("people") or []:
        add(str(person), str(person), "person")
    for org in custom.get("organisations") or []:
        add(str(org), str(org), "organisation")
    return names


def stored_urls() -> Dict[str, str]:
    """Normalised URL -> uuid of every note stored from a web address, in one scan of the table
    (the bookmarklet keeps a URL as given, tracking parameters and all)."""
    return {normalise_url(path): uuid for uuid, path in db.web_source_paths()}


def extract(note: NoteModel) -> List[dict]:
    """Links and names, each link marked with the note it already is (existing_uuid)."""
    links = extract_links(note)
    if links:
        stored = stored_urls()
        for entry in links:
            entry["existing_uuid"] = stored.get(normalise_url(entry["url"]))
    return links + extract_names(note)


_cache: Optional[JsonCache] = None
_cache_lock = threading.Lock()


def _lookup_cache() -> JsonCache:
    """Read once per process (the whole file was read and written on every lookup)."""
    global _cache
    with _cache_lock:
        path = settings.storage_path / "lookups" / "references.json"
        if _cache is None or _cache.path != path:
            _cache = JsonCache(path)
        return _cache


def _words(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def _matches(query: str, candidate: dict) -> bool:
    """More than half the query's words in the page's title or description: Wikipedia's search
    returns pages for anything, so a name without a page got pages sharing one word with it
    ("Keith Baar" -> "Judy Baar Topinka")."""
    wanted = _words(query)
    found = wanted & _words(f"{candidate['title']} {candidate['description']}")
    return len(found) * 2 > len(wanted)


def lookup(query: str, client: Optional[httpx.Client] = None) -> List[dict]:
    """Wikipedia pages for a name: [{title, url, description}], best match first (cached)."""
    if not query.strip():
        return []
    cache = _lookup_cache()
    if query in cache:
        return cache.get(query)
    own = client is None
    client = client or new_client()
    try:
        pages = search_pages(client, query, LOOKUP_LIMIT, prop="description|info", inprop="url")
    finally:
        if own:
            client.close()
    candidates = [{"title": p["title"], "url": p.get("fullurl") or "", "description": p.get("description", "")}
                  for p in pages if p.get("fullurl")]
    candidates = [c for c in candidates if _matches(query, c)]
    cache.set(query, candidates)
    cache.save()
    return candidates


def add_referrer(note: NoteModel, source: NoteModel) -> None:
    """source as one more note a stored page is referenced from (a link edge)."""
    from .linker import update_edges_for_note

    referenced_by = list(note.metadata.custom.get("referenced_by") or [])
    if source.uuid == note.uuid or source.uuid in referenced_by:
        return
    note.metadata.custom["referenced_by"] = referenced_by + [source.uuid]
    db.update_note_metadata(note.uuid, note.metadata.model_dump())
    if not settings.is_primary:  # the primary leaves linking to the worker's batch
        update_edges_for_note(note.uuid)


def referenced_note(note: NoteModel, source_uuid: str, source_title: str) -> NoteModel:
    """A fetched page as the note it becomes: the referrer named in its text, a link edge to it."""
    # plain text, not a [[wikilink]]: that resolves by title and would link any note of the same
    # name; the edge comes from referenced_by
    note.full_text = f"{note.full_text.rstrip()}\n\nReferenced from {source_title.strip()}\n"
    note.content_hash = compute_content_hash(note.full_text)
    note.metadata.custom["referenced_by"] = [source_uuid]
    return note


def fetch(source: NoteModel, url: str) -> ProcessingResult:
    """The page at url as a note referenced by source. A page already stored is returned as a
    duplicate, with source added to the notes it is referenced from."""
    from .parser import parse_url
    from .pipeline import ingest_note

    url = clean_url(url)
    existing_uuid = stored_urls().get(normalise_url(url))
    existing = db.get_note_by_uuid(existing_uuid) if existing_uuid else None
    if existing:
        add_referrer(existing, source)
        return ProcessingResult(status="duplicate", uuid=existing.uuid, title=existing.title)
    if settings.is_primary:  # the worker reads it (ingestqueue)
        from .ingestqueue import IngestQueue

        item = IngestQueue().add_url(url, referenced_by=source)
        return ProcessingResult(status="queued", uuid=item["placeholder"], title=item["title"])
    result = parse_url(url)
    if result.status != "success" or not result.note:
        return result
    note = referenced_note(result.note, source.uuid, source.title)
    ingest_note(note)
    return ProcessingResult(status="success", uuid=note.uuid, title=note.title, note=note)
