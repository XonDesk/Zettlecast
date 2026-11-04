"""
Reddit saves -> notes.

The saved list comes from the account's private feed (reddit.com/prefs/feeds, "your saved links";
REDDIT_SAVED_FEED in .env), so no Reddit app or login is needed. Reddit serves the newest ~1000
saves that way. Each save becomes one note (source_type "reddit"):

- a post: its text, the linked article (link posts, through the web parser) and the top comments
- a comment: the comment, the post it's on and the thread's top comments

The comments are curated (`curation.py`, LLM + the user's rubric): what's worth finding again
(products, advice, experience, insight) becomes one-line takeaways under "From the comments",
indexed; the raw comments stay on the note under "Thread comments", which isn't indexed.

Everything is read as RSS (Atom): Reddit sends anonymous JSON requests to its login page, but
still serves feeds. A thread feed has the post and its comments in Reddit's top order, without
scores or reply nesting. Feeds allow only about 10 requests a minute without a login, so the
client paces itself and follows Reddit's rate-limit headers. A thread that can't be fetched
leaves a note made from the saved feed alone; `refresh` tries it again.

Note ids derive from the Reddit id, so a refresh replaces the note in place. What was imported is
kept in <storage>/reddit_saved.json; a note deleted in the UI isn't imported again.
"""

import json
import logging
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse, urlunparse
from uuid import NAMESPACE_URL, uuid5

import httpx
from bs4 import BeautifulSoup

from . import runstate, statefile
from .config import Settings, settings
from .identity import compute_content_hash
from .models import NoteMetadata, NoteModel

logger = logging.getLogger(__name__)

USER_AGENT = "windows:zettlecast:0.1 (personal knowledge base; saved posts of the feed owner)"
REQUEST_INTERVAL = 6.5  # seconds between requests: Reddit allows ~10 a minute without a login
MAX_RETRIES = 4
THREAD_COMMENTS = 25  # top-level comments kept per thread (one request either way; curation picks)
RAW_HEADING = "Thread comments"  # unindexed (chunker.UNINDEXED_SECTIONS)
TAKEAWAYS_HEADING = "From the comments"
ARTICLE_CHARS = 20_000
POST_CHARS_IN_COMMENT_NOTE = 3_000
ATOM = "{http://www.w3.org/2005/Atom}"

# Links that are media or Reddit itself: nothing for the web parser to read
MEDIA_HOSTS = ("reddit.com", "redd.it", "imgur.com", "gfycat.com", "redgifs.com", "youtube.com",
               "youtu.be", "streamable.com", "giphy.com", "tenor.com")
MEDIA_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".gifv", ".webp", ".mp4", ".webm", ".mov")
GONE = {"[deleted]", "[removed]", ""}
BOTS = {"AutoModerator"}

_lock = threading.Lock()  # one sync at a time
_stop = threading.Event()  # set: the sync ends after the save it's on
progress: Dict[str, object] = {"running": False, "done": 0, "total": 0, "current": None}


class RedditError(Exception):
    pass


def state_path() -> Path:
    return settings.storage_path / "reddit_saved.json"


def _load() -> dict:
    try:
        data = json.loads(state_path().read_text(encoding="utf-8"))
        return {"imported": data.get("imported", {}), "last_sync": data.get("last_sync")}
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        logger.error(f"Reddit state unreadable, starting empty: {e}")
    return {"imported": {}, "last_sync": None}


def _save(state: dict) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = statefile.tmp_path(path)
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    for attempt in range(40):  # a reader may hold the file open for a moment on Windows
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.025)


def feed_url() -> str:
    """The configured feed, read from .env now (it can be set from the UI without a restart)."""
    return (Settings().reddit_saved_feed or "").strip()


def rss_feed(url: str) -> str:
    """The RSS form of a private saved feed link (the JSON link on the feeds page works too)."""
    parts = urlparse(url.strip())
    if not parts.netloc.endswith("reddit.com") or "feed" not in parse_qs(parts.query) or "/saved" not in parts.path:
        raise RedditError("Not a private saved feed link: copy the RSS link for your saved links "
                          "from reddit.com/prefs/feeds")
    path = parts.path.rstrip("/")
    for suffix in ("/.rss", ".rss", "/.json", ".json"):
        path = path.removesuffix(suffix)
    return urlunparse(parts._replace(scheme="https", netloc="www.reddit.com", path=f"{path}.rss"))


# --- HTTP ---


class Client:
    """Paced GETs: one request per interval, longer when Reddit's headers say so, waits on 429."""

    def __init__(self, interval: float = REQUEST_INTERVAL, sleep: Callable[[float], None] = time.sleep):
        self.interval = interval
        self.sleep = sleep
        self.next_at = 0.0
        self.http = httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30)

    def get_feed(self, url: str, params: Optional[dict] = None) -> List[dict]:
        """A feed's entries (see `entry`)."""
        for _ in range(MAX_RETRIES):
            wait = self.next_at - time.monotonic()
            if wait > 0:
                self.sleep(wait)
            # merged into the link's own query: httpx's params= would replace it (and drop feed=/user=)
            response = self.http.get(httpx.URL(url).copy_merge_params(params or {}))
            self.next_at = time.monotonic() + self._interval(response)
            if response.status_code == 429:
                self.sleep(float(response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset")
                                 or 60) + 1)
                continue
            if response.status_code != 200:
                raise RedditError(f"HTTP {response.status_code} for {urlparse(url).path}")
            try:
                return [entry(e) for e in ET.fromstring(response.text).iter(f"{ATOM}entry")]
            except ET.ParseError:
                raise RedditError(f"Not a feed from {urlparse(url).path} (Reddit may be blocking requests)")
        raise RedditError("Reddit kept answering 429 (too many requests)")

    def _interval(self, response: httpx.Response) -> float:
        try:
            remaining = float(response.headers["x-ratelimit-remaining"])
            reset = float(response.headers["x-ratelimit-reset"])
        except (KeyError, ValueError):
            return self.interval
        # spread what's left of the window over its remaining seconds
        return reset + 1 if remaining < 1 else max(self.interval, reset / remaining)

    def close(self) -> None:
        self.http.close()


def _text(html: str) -> str:
    """Reddit's rendered markdown as plain text: one paragraph per block, links as "text (url)"."""
    soup = BeautifulSoup(html or "", "html.parser")
    for a in soup.find_all("a"):
        href, label = a.get("href") or "", a.get_text()
        if href.startswith("http") and href != label.strip():
            a.replace_with(f"{label} ({href})")
    blocks = soup.find_all(["p", "li", "pre", "h1", "h2", "h3", "h4", "h5", "h6"])
    if not blocks:
        return soup.get_text(" ").strip()
    out = []
    for block in blocks:
        text = block.get_text().strip()
        if not text:
            continue
        prefix = "> " if block.find_parent("blockquote") else "- " if block.name == "li" else ""
        out.append(prefix + text)
    return "\n\n".join(out)


def entry(e: ET.Element) -> dict:
    """One Atom entry: id (t3_/t1_ fullname), title, author, subreddit, permalink, date, text, link."""
    def find(tag):
        return e.find(f"{ATOM}{tag}")

    html = (find("content").text if find("content") is not None else "") or ""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("div", class_="md")
    link = next((a.get("href") for a in soup.find_all("a") if a.get_text().strip() == "[link]"), None)
    if link and urlparse(link).netloc.lower().removeprefix("www.").endswith("reddit.com"):
        link = None  # a self post links to itself
    author = find(f"author/{ATOM}name")
    category = find("category")
    date = find("published") if find("published") is not None else find("updated")
    return {
        "name": find("id").text if find("id") is not None else "",
        "title": find("title").text if find("title") is not None else "",
        "author": (author.text or "").strip().removeprefix("/u/") if author is not None else "[deleted]",
        "subreddit": category.get("term") if category is not None else None,
        "permalink": find("link").get("href") if find("link") is not None else "",
        "date": (date.text or "")[:19] if date is not None else None,
        "text": _text(str(body)) if body else ("" if find("id") is not None and find("id").text.startswith("t3_")
                                               else _text(html)),
        "link": link,
    }


def fetch_saved(client: Client, url: str, limit: Optional[int] = None) -> List[dict]:
    """Every save in the feed, newest first."""
    items: List[dict] = []
    after = None
    while True:
        page = client.get_feed(url, {"limit": 100, **({"after": after} if after else {})})
        page = [i for i in page if i["name"].startswith(("t1_", "t3_"))]
        items += page
        if len(page) < 100 or (limit and len(items) >= limit):
            return items[:limit] if limit else items
        after = page[-1]["name"]


# --- Formatting ---


def _usable(c: dict) -> bool:
    return c["text"].strip() not in GONE and c["author"] not in BOTS and c["author"] != "[deleted]"


def thread_comments(comments: List[dict], skip: str = "") -> List[dict]:
    """The thread's first usable comments (the feed is in Reddit's top order): {author, text}."""
    picked = [c for c in comments if _usable(c) and c["name"] != skip][:THREAD_COMMENTS]
    return [{"author": c["author"], "text": c["text"]} for c in picked]


def comments_section(comments: List[dict]) -> str:
    return f"## {RAW_HEADING}\n" + "\n\n".join(f"**u/{c['author']}**\n{c['text']}" for c in comments)


def _linked_article(url: str) -> Optional[dict]:
    """The page a link post points to, through the web parser (None for media and Reddit links)."""
    parts = urlparse(url)
    host = parts.netloc.lower()
    if not host or any(host == h or host.endswith("." + h) for h in MEDIA_HOSTS):
        return None
    if parts.path.lower().endswith(MEDIA_EXTENSIONS):
        return None
    from .parser import parse_url

    result = parse_url(url, enrich=False)
    if result.status != "success" or not result.note:
        logger.info(f"Linked article not read ({result.error_message}): {url}")
        return None
    return {"url": url, "title": result.note.title, "text": result.note.full_text[:ARTICLE_CHARS]}


def _post_title(item: dict) -> str:
    """A comment entry's title is "/u/name on <post title>"."""
    prefix = f"/u/{item['author']} on "
    return item["title"].removeprefix(prefix) if item["name"].startswith("t1_") else item["title"]


def post_note_text(post: dict, comments: List[dict], article: Optional[dict]) -> str:
    """comments: from `thread_comments`."""
    parts = [f"r/{post['subreddit']} · posted by u/{post['author']} · {(post['date'] or '')[:10]}"]
    if post.get("link"):
        parts.append(f"Link: {post['link']}")
    if post["text"].strip() not in GONE:
        parts.append(f"## Post\n{post['text']}")
    if article:
        parts.append(f"## Linked article: {article['title'] or article['url']}\n{article['text']}")
    if comments:
        parts.append(comments_section(comments))
    return "\n\n".join(parts)


def comment_note_text(saved: dict, post: Optional[dict], comments: List[dict]) -> str:
    parts = [f"r/{saved['subreddit']} · comment by u/{saved['author']} · {(saved['date'] or '')[:10]}",
             f"On the post “{_post_title(saved)}”" + (f" by u/{post['author']}" if post else ""),
             f"## Saved comment\n{saved['text']}"]
    if post and post["text"].strip() not in GONE:
        text = post["text"]
        cut = text[:POST_CHARS_IN_COMMENT_NOTE] + ("…" if len(text) > POST_CHARS_IN_COMMENT_NOTE else "")
        parts.append(f"## The post\n{cut}")
    if comments:
        parts.append(comments_section(comments))
    return "\n\n".join(parts)


# --- Notes ---


def note_uuid(fullname: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"https://www.reddit.com/{fullname}"))


def thread_url(item: dict) -> str:
    """The post's comments feed (for a comment: its post's, the permalink minus the comment id)."""
    path = urlparse(item["permalink"]).path.rstrip("/")
    if item["name"].startswith("t1_"):
        path = path.rsplit("/", 1)[0]
    return f"https://www.reddit.com{path}/.rss"


def build_note(client: Optional[Client], item: dict, articles: bool = True) -> NoteModel:
    """A note for one save; with no client (or a failed fetch) only the saved feed's entry is used."""
    post, comments, fetched = None, [], False
    if client:
        try:
            # depth 1: top-level comments only (the feed has no reply nesting to tell them apart)
            thread = client.get_feed(thread_url(item),
                                     {"sort": "top", "limit": THREAD_COMMENTS + 10, "depth": 1})
            post = next((e for e in thread if e["name"].startswith("t3_")), None)
            comments = [e for e in thread if e["name"].startswith("t1_")]
            fetched = True
        except (RedditError, httpx.HTTPError) as e:
            logger.warning(f"Thread not fetched for {item['permalink']}: {e}")

    kept = thread_comments(comments, skip=item["name"])
    if item["name"].startswith("t3_"):
        post = {**item, **(post or {})}
        article = _linked_article(post["link"]) if articles and post.get("link") else None
        text = post_note_text(post, kept, article)
        title = post["title"] or "Reddit post"
    else:
        text = comment_note_text(item, post, kept)
        title = f"Comment by u/{item['author']}: {_post_title(item)}"

    return NoteModel(
        uuid=note_uuid(item["name"]),
        title=title[:300],
        source_type="reddit",
        source_path=item["permalink"],
        full_text=text,
        content_hash=compute_content_hash(text),
        metadata=NoteMetadata(
            source_url=item["permalink"],
            author=item["author"],
            word_count=len(text.split()),
            custom={"reddit": {
                "kind": "post" if item["name"].startswith("t3_") else "comment",
                "fullname": item["name"],
                "subreddit": item["subreddit"],
                "created": item["date"],
                **({"link_url": item["link"]} if item.get("link") else {}),
                "post_title": _post_title(item),
                "thread_fetched": fetched,
                "comments": kept,  # raw, for curation (again, after a rubric change) without asking Reddit
                "curated": False,
            }},
        ),
    )


def request_stop() -> None:
    """Ask a running sync to stop after its current save (what's imported so far stays)."""
    _stop.set()


def sync(limit: Optional[int] = None, refresh: bool = False, articles: bool = True, enrich: bool = True,
         threads: bool = True, client: Optional[Client] = None,
         on_item: Optional[Callable[[int, int, str], None]] = None) -> dict:
    """
    Import the saves not imported yet (refresh: also re-fetch those whose thread failed before),
    newest first. Returns counts.

    enrich=False leaves the LLM out (no tags or summary; nothing loaded on the GPU but the embedder):
    those notes are tagged later from the database by `tag_untagged`, which a sync with the LLM runs
    at its end, so Reddit isn't asked for them again.

    threads=False builds each note from the saved feed alone (no request per save: Reddit allows
    anonymous clients about one a minute); a later sync with refresh=True adds the threads' comments.
    """
    from .db import db
    from .enricher import enrich_note

    enrich = enrich and not settings.is_primary  # no LLM on the primary: the worker tags saves later
    url = feed_url()
    if not url:
        raise RedditError("No feed configured: set REDDIT_SAVED_FEED (Settings page, or .env) to the "
                          "RSS link for your saved links on reddit.com/prefs/feeds")
    if not _lock.acquire(blocking=False):
        raise RedditError("A Reddit sync is already running")
    own_client = client is None
    client = client or Client()
    counts = {"saved": 0, "imported": 0, "refreshed": 0, "partial": 0, "unenriched": 0, "tagged": 0,
              "failed": 0, "stopped": False}
    _stop.clear()
    try:
        progress.update(running=True, done=0, total=0, current="reading the saved feed", done_times=[],
                        phase="sync")
        items = fetch_saved(client, rss_feed(url))
        counts["saved"] = len(items)
        state = _load()
        todo = [i for i in items if i["name"] not in state["imported"]
                or (refresh and not state["imported"][i["name"]].get("thread"))]
        todo = todo[:limit] if limit else todo
        progress.update(total=len(todo))
        for n, item in enumerate(todo, 1):
            if _stop.is_set():
                counts["stopped"] = True
                break
            label = _post_title(item) or item["name"]
            if n > 1:
                runstate.record_done(progress)
            progress.update(done=n - 1, current=label)
            if on_item:
                on_item(n, len(todo), label)
            again = item["name"] in state["imported"]
            try:
                note = build_note(client if threads else None, item, articles=articles)
                if enrich:
                    if fetched_thread(note):
                        curate_note(note)  # before tagging: tags come from the takeaways, not the banter
                    note = enrich_note(note)
                elif again:
                    _keep_tags(note)
                db.upsert_note(note)  # linked by one graph rebuild at the end (~10 s per note one by one)
            except Exception as e:  # one bad save shouldn't end the sync
                logger.exception(f"Reddit save {item['name']} failed: {e}")
                counts["failed"] += 1
                continue
            fetched = fetched_thread(note)
            enriched = "summary" in note.metadata.custom
            counts["refreshed" if again else "imported"] += 1
            counts["partial"] += not fetched
            counts["unenriched"] += not enriched
            state["imported"][item["name"]] = {"uuid": note.uuid, "title": note.title, "thread": fetched,
                                               "enriched": enriched,
                                               "curated": bool(note.metadata.custom["reddit"].get("curated")),
                                               "at": datetime.now().isoformat(timespec="seconds")}
            _save(state)
        if not counts["stopped"]:
            progress.update(done=len(todo))
            state["last_sync"] = datetime.now().isoformat(timespec="seconds")
            _save(state)
            if enrich:
                counts["tagged"] = _tag_untagged(on_item)["tagged"]
        return counts
    finally:
        try:
            # the primary stores saves waiting for the worker, whose batch links them
            if (counts["imported"] or counts["refreshed"] or counts["tagged"]) and not settings.is_primary:
                progress.update(current="linking notes into the graph")
                _relink()
        finally:
            progress.update(running=False, current=None)
            if own_client:
                client.close()
            _stop.clear()
            _lock.release()


def _relink() -> None:
    """Edges + communities for everything stored by a sync (one rebuild: per note it's ~10 s each)."""
    from .linker import rebuild_graph

    try:
        rebuild_graph()
    except Exception as e:  # the notes are stored; the next rebuild links them
        logger.exception(f"Graph rebuild after the Reddit sync failed: {e}")


def _keep_tags(note: NoteModel) -> None:
    """A save rebuilt without the LLM keeps the tags and summary its stored note already has."""
    from .db import db

    stored = db.get_note_by_uuid(note.uuid)
    if stored is None or "summary" not in stored.metadata.custom:
        return
    note.metadata.tags = stored.metadata.tags
    note.metadata.custom = {**stored.metadata.custom, **note.metadata.custom}


def tag_untagged(on_item: Optional[Callable[[int, int, str], None]] = None) -> dict:
    """
    LLM work left on imported saves, from the stored notes (no Reddit requests): comment curation for
    saves that have their thread but no takeaways yet, then tags + summary.
    """
    if not _lock.acquire(blocking=False):
        raise RedditError("A Reddit sync is already running")
    _stop.clear()
    counts = {}
    try:
        counts = _tag_untagged(on_item)
        return counts
    finally:
        if counts.get("tagged"):
            progress.update(current="linking notes into the graph")
            _relink()
        progress.update(running=False, current=None)
        _stop.clear()
        _lock.release()


def _tag_untagged(on_item: Optional[Callable[[int, int, str], None]] = None) -> dict:
    from .db import db

    state = _load()
    todo = [(name, entry) for name, entry in state["imported"].items() if _needs_llm(entry)]
    counts = {"tagged": 0, "curated": 0, "failed": 0, "missing": 0, "stopped": False}
    progress.update(running=True, done=0, total=len(todo), current="tagging saves", done_times=[], phase="tag")
    for n, (name, entry) in enumerate(todo, 1):
        if _stop.is_set():
            counts["stopped"] = True
            break
        if n > 1:
            runstate.record_done(progress)
        progress.update(done=n - 1, current=f"tagging: {entry.get('title', name)}")
        if on_item:
            on_item(n, len(todo), entry.get("title", name))
        note = db.get_note_by_uuid(entry["uuid"])
        if note is None:  # deleted in the UI: leave it deleted
            entry["enriched"] = None
            counts["missing"] += 1
            _save(state)
            continue
        note, curated = llm_pass(note)
        if note is None:  # LLM unavailable: stop, the rest would fail too
            counts["failed"] += 1
            break
        counts["curated"] += curated
        db.upsert_note(note)
        entry["enriched"] = True
        entry["curated"] = bool(note.metadata.custom["reddit"].get("curated"))
        counts["tagged"] += 1
        _save(state)
    progress.update(done=counts["tagged"] + counts["missing"])
    return counts


def llm_pass(note: NoteModel) -> tuple:
    """
    The LLM work on one save: its comments curated (thread fetched, no takeaways yet), then tags +
    summary. Returns (note, whether it was curated now); the note is None when the LLM couldn't.
    A save with too little text for the LLM (an image or link post: its title) is done as it is.
    """
    from .enricher import enrich_note, too_short
    from .llm import LLMError

    curated = False
    if fetched_thread(note) and not note.metadata.custom["reddit"].get("curated"):
        if not curate_note(note):
            return None, False
        curated = True
        # tags made from the raw comments: again from the takeaways
        note.metadata.tags = []
        for key in ("summary", "people", "organisations"):
            note.metadata.custom.pop(key, None)
    if too_short(note.full_text):
        note.metadata.custom.setdefault("summary", "")
        return note, curated
    try:
        note = enrich_note(note, raise_errors=True)
    except LLMError as e:
        logger.warning(f"Save not tagged: {note.title}: {e}")
        return None, curated
    return note, curated


def needs_llm(note_meta: dict) -> bool:
    """Whether a stored save (its metadata) still waits for curation or tags."""
    custom = note_meta.get("custom") or {}
    reddit = custom.get("reddit") or {}
    return "summary" not in custom or (bool(reddit.get("thread_fetched")) and not reddit.get("curated"))


def mark_tagged(note: NoteModel) -> None:
    """Record in the import state that a save got its LLM pass (skipped while a sync holds the state)."""
    if not _lock.acquire(blocking=False):
        return  # a sync rewrites the state; status counts catch up with its next save
    try:
        state = _load()
        for entry in state["imported"].values():
            if entry.get("uuid") == note.uuid:
                entry["enriched"] = True
                entry["curated"] = bool(note.metadata.custom["reddit"].get("curated"))
                _save(state)
                return
    finally:
        _lock.release()


def _needs_llm(entry: dict) -> bool:
    if entry.get("enriched") is None:  # deleted in the UI
        return False
    return entry.get("enriched") is False or (bool(entry.get("thread")) and not entry.get("curated"))


# --- Comment curation ---


def fetched_thread(note: NoteModel) -> bool:
    return bool(note.metadata.custom["reddit"].get("thread_fetched"))


def _split_sections(text: str) -> List[str]:
    return re.split(r"(?m)^(?=## )", text)


def _section(text: str, *titles: str) -> str:
    for part in _split_sections(text):
        head, _, body = part.partition("\n")
        if head.removeprefix("## ").strip() in titles or any(head.startswith(f"## {t}") for t in titles):
            return body.strip()
    return ""


OLD_COMMENT_HEADINGS = ("Top comments", "Other top comments")  # notes imported before curation


def _stored_comments(note: NoteModel) -> List[dict]:
    """The note's raw comments; an older note has them only in its text (moved to the raw section here)."""
    meta = note.metadata.custom["reddit"]
    if meta.get("comments") is not None:
        return meta["comments"]
    body = _section(note.full_text, *OLD_COMMENT_HEADINGS)
    pieces = re.split(r"(?m)^\*\*u/([^*\n]+)\*\*\n", body)
    comments = [{"author": a.strip(), "text": t.strip()}
                for a, t in zip(pieces[1::2], pieces[2::2]) if t.strip()]
    parts = [p for p in _split_sections(note.full_text)
             if not any(p.startswith(f"## {h}") for h in OLD_COMMENT_HEADINGS)]
    note.full_text = "".join(parts).rstrip() + ("\n\n" + comments_section(comments) if comments else "")
    meta["comments"] = comments
    return comments


def _curation_input(note: NoteModel) -> dict:
    meta = note.metadata.custom["reddit"]
    text = note.full_text
    post = _section(text, "Post", "The post")
    article = _section(text, "Linked article")
    if article:
        post += f"\n\nLinked article (start):\n{article[:1500]}"
    saved = None
    if meta.get("kind") == "comment":
        saved = {"author": note.metadata.author or "", "text": _section(text, "Saved comment")}
    title = meta.get("post_title") or note.title.split(": ", 1)[-1]
    return {"title": title, "subreddit": meta.get("subreddit") or "", "post": post,
            "comments": _stored_comments(note), "saved": saved}


def _with_takeaways(text: str, takeaways: str) -> str:
    """The note text with its takeaways section replaced, placed before the raw comments."""
    parts = [p for p in _split_sections(text) if not p.startswith(f"## {TAKEAWAYS_HEADING}")]
    if not takeaways:
        return "".join(parts).rstrip()
    at = next((i for i, p in enumerate(parts) if p.startswith(f"## {RAW_HEADING}")), len(parts))
    parts = [p.rstrip() + "\n\n" for p in parts]
    parts.insert(at, takeaways + "\n\n")
    return "".join(parts).rstrip()


def curate_note(note: NoteModel) -> bool:
    """Takeaways from the note's comments into its text (in place). False if the LLM couldn't do it."""
    from .curation import curate, section
    from .llm import LLMError

    meta = note.metadata.custom["reddit"]
    args = _curation_input(note)
    try:
        verdicts = curate(**args)
    except LLMError as e:
        logger.warning(f"Comments not curated for {note.title}: {e}")
        return False
    note.full_text = _with_takeaways(note.full_text, section(verdicts, TAKEAWAYS_HEADING))
    note.content_hash = compute_content_hash(note.full_text)
    note.metadata.word_count = len(note.full_text.split())
    meta["curated"] = True
    meta["takeaways"] = sum(v["keep"] for v in verdicts)
    return True


def review_sample(count: int = 20, seed: int = 1) -> List[dict]:
    """
    Curate a sample of imported saves that have comments, without storing anything: each save's
    verdicts with the comments they're about, for the user to check against their own judgement.
    """
    from .curation import curate
    from .db import db

    state = _load()
    names = sorted(name for name, entry in state["imported"].items()
                   if entry.get("thread") and entry.get("enriched") is not None)
    random.Random(seed).shuffle(names)
    out = []
    for name in names:
        if len(out) >= count:
            break
        note = db.get_note_by_uuid(state["imported"][name]["uuid"])
        if note is None:
            continue
        args = _curation_input(note)
        if not args["comments"]:
            continue
        progress.update(current=f"reviewing: {note.title}")
        verdicts = curate(**args)
        out.append({"title": note.title, "url": note.metadata.source_url,
                    "kind": note.metadata.custom["reddit"]["kind"], **args, "verdicts": verdicts})
    return out


def status() -> dict:
    state = _load()
    return {"configured": bool(feed_url()), "imported": len(state["imported"]),
            "partial": sum(1 for v in state["imported"].values() if not v.get("thread")),
            "untagged": sum(1 for v in state["imported"].values() if v.get("enriched") is False),
            "uncurated": sum(1 for v in state["imported"].values()
                             if v.get("thread") and not v.get("curated") and v.get("enriched") is not None),
            "last_sync": state["last_sync"], "progress": dict(progress)}
