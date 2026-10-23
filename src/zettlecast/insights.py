"""
Today's insights: a short daily set of graph connections worth a second look.

Cards are graph links chosen for surprise, not strength alone: across communities, across source
types (a podcast and a screenshot), across years, or the same idea in other words (similar
passages, few shared names). Echoes pair something recent (published, posted or taken lately: not
merely imported, a Reddit import brings years of saves at once) with something much older. A pair
is shown once; each note at most once a day; near-duplicates never.

Cards are picked ahead (UPCOMING), each with its material (titles, the matching passages or the
notes' briefs), so the LLM can prepare them before the day they're shown: it says what links the
two, asks a question, and how they relate (agree, contradict, extend, apply). A pair it finds
unrelated is not shown. Standalone, the LLM runs in the background when the page asks; on a
primary, the worker claims cards like tag jobs (worker_api "insight").

Verdicts: good (kept), rejected (the link leaves the graph: db.add_rejected_edge), noted (a note
was written linking both). State: <storage>/insights.json.
"""

import json
import logging
import re
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from . import statefile
from .config import settings

logger = logging.getLogger(__name__)

PER_DAY = 5
UPCOMING = 15  # cards picked ahead, so the LLM can prepare the next days' sets
MAX_ECHOES = 2  # per day; the rest are surprising links
ECHO_NEW_DAYS = 14  # an echo's new note is from this recently (its own date)...
ECHO_GAP_DAYS = 180  # ...and its partner at least this much older
ECHO_BONUS = 0.3  # an echo outranks a link that is otherwise as surprising
NEAR_DUPLICATE = 0.9  # vector similarity: the same thing twice (a repost, two episodes' shared intro)
RECENT_NOTE_DAYS = 7  # a note shown on a card isn't shown again this soon
MAX_ATTEMPTS = 3  # LLM tries per card
RELATIONS = ("agrees", "contradicts", "extends", "applies", "unrelated")

_lock = threading.RLock()
_claimed: Dict[str, float] = {}  # key -> lease expiry (monotonic): with the worker
_generating = threading.Event()


def _path() -> Path:
    return settings.storage_path / "insights.json"


def _load() -> dict:
    data = statefile.read(_path())
    data.setdefault("cards", {})
    data.setdefault("days", {})
    return data


def _save(data: dict) -> None:
    statefile.write(_path(), data)


def pair_key(a: str, b: str) -> str:
    return "_".join(sorted((a, b)))


def _today() -> str:
    return date.today().isoformat()


# --- Choosing pairs ---


def _note_date(created_at: str, custom: dict) -> Optional[datetime]:
    """When a note's content is from: the episode's or post's date, else when it was added."""
    from .brief import iso_date

    day = (iso_date((custom.get("podcast") or {}).get("published"))
           or iso_date((custom.get("reddit") or {}).get("created")) or iso_date(created_at))
    try:
        return datetime.fromisoformat(day) if day else None
    except ValueError:
        return None


def _notes() -> Dict[str, dict]:
    """Every note: title, type, source group, community, dates, brief, image (what cards need)."""
    from .db import db
    from .linker import source_group
    from .brief import has_image, note_brief

    community = {u: int(c["community"]) for c in db.get_communities() for u in c["members"]}
    labels = {int(c["community"]): c["label"] for c in db.get_communities()}
    notes = {}
    for r in (db.notes.search().select(["uuid", "title", "source_type", "source_path", "created_at", "metadata_json"])
              .limit(1_000_000).to_list()):
        meta = json.loads(r["metadata_json"] or "{}")
        custom = meta.get("custom") or {}
        c = community.get(r["uuid"])
        notes[r["uuid"]] = {
            "uuid": r["uuid"], "title": r["title"], "source_type": r["source_type"],
            "group": source_group({"source_type": r["source_type"], "source_path": r["source_path"], "metadata": meta}),
            "community": c, "community_label": labels.get(c),
            "date": _note_date(r["created_at"], custom),
            "brief": note_brief(custom), "image": has_image(r["source_type"], custom),
        }
    return notes


def surprise(edge: dict, a: dict, b: dict, shared_people: int = 0) -> float:
    """
    How unexpected a link is, about 0..1: different communities, different kinds of source,
    years apart, and the same idea in other words (matching passages, few shared names). A
    person in both is what you'd expect (the same guest on the same show), so it counts against.
    """
    score = 0.0
    if a["community"] is not None and b["community"] is not None:
        score += 0.3 if a["community"] != b["community"] else 0.0
    else:
        score += 0.1
    if a["source_type"] != b["source_type"]:
        score += 0.3
    elif a["group"] != b["group"]:
        score += 0.15  # another show, site or subreddit
    if a["date"] and b["date"]:
        score += 0.15 * min(abs((a["date"] - b["date"]).days) / 730, 1.0)
    if edge.get("passage_sim", 0) > 0:
        score += 0.25 * (1.0 - min(edge.get("entity_sim", 0) / 0.3, 1.0))
    return score - 0.2 * min(shared_people, 1)


def candidates(notes: Dict[str, dict], edges: List[dict], exclude: set, now: Optional[datetime] = None) -> List[dict]:
    """Unseen links worth a card, best first: {key, kind: echo | surprise, score, a, b, edge}."""
    now = now or datetime.now()
    similar = [e for e in edges if (e.get("kind") or "similar") == "similar"]
    if not similar:
        return []
    weights = sorted(e["weight"] for e in similar)
    floor = weights[len(weights) // 2]  # the stronger half only: surprise among weak links is noise
    out = []
    for e in similar:
        a, b = notes.get(e["source_uuid"]), notes.get(e["target_uuid"])
        key = pair_key(e["source_uuid"], e["target_uuid"])
        if a is None or b is None or key in exclude or e["weight"] < floor or e.get("vector_sim", 0) >= NEAR_DUPLICATE:
            continue
        new, old = (a, b) if (a["date"] or datetime.min) >= (b["date"] or datetime.min) else (b, a)
        echo = bool(new["date"] and old["date"] and now - new["date"] <= timedelta(days=ECHO_NEW_DAYS)
                    and (new["date"] - old["date"]).days >= ECHO_GAP_DAYS)
        shared = json.loads(e.get("reason_json") or "{}").get("shared", [])
        # surprise first; strength (already in the stronger half) breaks ties among the surprising
        score = surprise(e, a, b, sum(1 for s in shared if s.get("kind") == "person")) + 0.5 * e["weight"]
        if echo:
            out.append({"key": key, "kind": "echo", "score": score + ECHO_BONUS, "a": new, "b": old, "edge": e})
        else:
            out.append({"key": key, "kind": "surprise", "score": score, "a": a, "b": b, "edge": e})
    out.sort(key=lambda c: -c["score"])
    return out


def _side(note: dict, passage: Optional[dict]) -> dict:
    """One note on a card: what it is, and the passage that matches the other note (or its brief)."""
    if passage:
        # plain text: no leading "[12.3s - 45.6s]" (the time is kept apart), headings or bold marks
        text = re.sub(r"^\s*\[[\d.]+s(?:\s*-\s*[\d.]+s)?\]\s*", "", passage["text"])
        text = re.sub(r"(?m)^#+ .*$", "", text).replace("**", "")
        text = re.sub(r"\n{2,}", "\n", text).strip()
    else:
        text = note["brief"]
    return {
        "uuid": note["uuid"], "title": note["title"], "source_type": note["source_type"],
        "community": note["community_label"], "date": note["date"].date().isoformat() if note["date"] else None,
        "text": text, "time": passage.get("time") if passage else None, "passage": bool(passage),
        "image": note["image"],
    }


def _card(c: dict) -> dict:
    """A chosen pair as stored: self-contained, so it can be shown or prepared without the database."""
    reason = json.loads(c["edge"].get("reason_json") or "{}")
    pair = reason.get("passage") or {}
    side_of = {c["edge"]["source_uuid"]: pair.get("source"), c["edge"]["target_uuid"]: pair.get("target")}
    return {
        "key": c["key"], "kind": c["kind"], "score": round(c["score"], 3),
        "weight": round(float(c["edge"]["weight"]), 3),
        "shared": [s["name"] for s in reason.get("shared", [])][:6],
        "a": _side(c["a"], side_of.get(c["a"]["uuid"])),
        "b": _side(c["b"], side_of.get(c["b"]["uuid"])),
        "picked": _today(), "insight": None, "attempts": 0, "verdict": None,
    }


def _scheduled(data: dict) -> set:
    return {k for keys in data["days"].values() for k in keys}


def refill(data: dict, notes: Optional[Dict[str, dict]] = None) -> int:
    """Pick cards ahead until UPCOMING wait (not on a day, not decided). Returns how many were added."""
    from .db import db

    upcoming = [k for k, c in data["cards"].items() if c["verdict"] is None and k not in _scheduled(data)
                and (c["insight"] or {}).get("relation") != "unrelated"]
    need = UPCOMING - len(upcoming)
    if need <= 0:
        return 0
    notes = notes if notes is not None else _notes()
    in_use = {c[side]["uuid"] for k in upcoming for side in ("a", "b") for c in [data["cards"][k]]}
    # echoes in the same share as a day takes them, or the queue fills with echoes a day can't use
    echoes = sum(1 for k in upcoming if data["cards"][k]["kind"] == "echo")
    max_echoes = UPCOMING * MAX_ECHOES // PER_DAY
    added = 0
    for c in candidates(notes, db.get_all_edges(), exclude=set(data["cards"])):
        if added >= need:
            break
        if c["a"]["uuid"] in in_use or c["b"]["uuid"] in in_use:
            continue  # variety: a note waits on one card at a time
        if c["kind"] == "echo" and echoes >= max_echoes:
            continue
        data["cards"][c["key"]] = _card(c)
        in_use |= {c["a"]["uuid"], c["b"]["uuid"]}
        echoes += c["kind"] == "echo"
        added += 1
    return added


def _recent_notes(data: dict, today: str) -> set:
    since = (date.fromisoformat(today) - timedelta(days=RECENT_NOTE_DAYS)).isoformat()
    return {data["cards"][k][side]["uuid"] for day, keys in data["days"].items() if since <= day < today
            for k in keys if k in data["cards"] for side in ("a", "b")}


def _unrelated(card: dict) -> bool:
    return (card["insight"] or {}).get("relation") == "unrelated"


def plan_day(data: dict, today: str, keep: Optional[List[str]] = None) -> List[str]:
    """
    Today's set from the cards waiting: prepared ones first, at most MAX_ECHOES echoes, no note
    twice. keep: cards already on today's set (topping it up after the LLM found one unrelated).
    """
    scheduled = _scheduled(data)
    waiting = [c for k, c in data["cards"].items() if c["verdict"] is None and k not in scheduled
               and not _unrelated(c)]
    waiting.sort(key=lambda c: (c["insight"] is None, -c["score"]))
    recent = _recent_notes(data, today)
    chosen = list(keep or [])
    used = {data["cards"][k][side]["uuid"] for k in chosen for side in ("a", "b")}
    echoes = sum(1 for k in chosen if data["cards"][k]["kind"] == "echo")
    for strict in (True, False):  # notes from the last days' cards only if nothing else is left
        for c in waiting:
            if len(chosen) >= PER_DAY:
                break
            ids = {c["a"]["uuid"], c["b"]["uuid"]}
            if c["key"] in chosen or ids & used or (strict and ids & recent):
                continue
            if c["kind"] == "echo" and echoes >= MAX_ECHOES:
                continue
            chosen.append(c["key"])
            used |= ids
            echoes += c["kind"] == "echo"
    return chosen


def today() -> dict:
    """Today's cards (planned on the first call of the day), with how many still wait for the LLM."""
    from .db import db

    day = _today()
    with _lock:
        data = _load()
        if day not in data["days"]:
            notes = _notes()
            refill(data, notes)
            data["days"][day] = plan_day(data, day)
            refill(data, notes)  # the next days' cards, so the LLM can prepare them
            _save(data)
        keys = [k for k in data["days"][day] if k in data["cards"]]
        kept = [k for k in keys if not _unrelated(data["cards"][k])]
        added = [k for k in plan_day(data, day, kept) if k not in keys] if len(kept) < PER_DAY else []
        if added:
            # the LLM found a card's pair unrelated: another card takes its place
            data["days"][day] = keys + added
            refill(data)
            _save(data)
        cards = [data["cards"][k] for k in data["days"][day]]
    alive = set(db.note_titles([u for c in cards for u in (c["a"]["uuid"], c["b"]["uuid"])]))
    shown = [c for c in cards if c["a"]["uuid"] in alive and c["b"]["uuid"] in alive and not _unrelated(c)]
    return {"date": day, "cards": shown, "preparing": sum(1 for c in shown if c["insight"] is None),
            "generating": _generating.is_set()}


# --- Verdicts ---


def decide(key: str, verdict: str, note_uuid: Optional[str] = None) -> dict:
    """good | rejected (the pair is unlinked for good) | noted (a note was written from it)."""
    from .db import db

    if verdict not in ("good", "rejected", "noted"):
        raise ValueError(f"Unknown verdict {verdict!r}")
    with _lock:
        data = _load()
        card = data["cards"].get(key)
        if card is None:
            raise KeyError(key)
        card.update(verdict=verdict, decided_at=datetime.now().isoformat(timespec="seconds"))
        if note_uuid:
            card["note"] = note_uuid
        _save(data)
    if verdict == "rejected":
        db.add_rejected_edge(card["a"]["uuid"], card["b"]["uuid"])
    return card


def write_note(key: str, title: str, text: str) -> str:
    """A note of the user's own from a card: markdown in <storage>/notes, [[linked]] to both notes."""
    from .identity import create_note_file, generate_uuid
    from .parser import parse_markdown
    from .pipeline import ingest_note

    with _lock:
        card = _load()["cards"].get(key)
    if card is None:
        raise KeyError(key)
    title = title.strip() or f"{card['a']['title']} × {card['b']['title']}"
    body = text.strip()
    for side in ("a", "b"):
        link = f"[[{card[side]['title']}]]"
        if link not in body:
            body += f"\n\n{link}" if side == "a" else f" {link}"
    uuid = generate_uuid()
    path = create_note_file(uuid, title, body.strip() + "\n", "markdown", "", {"from_insight": key})
    target = settings.storage_path / "notes" / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    path.replace(target)
    result = parse_markdown(target)
    if result.status != "success" or result.note is None:
        raise RuntimeError(result.error_message or "could not read the new note")
    ingest_note(result.note)
    decide(key, "noted", result.note.uuid)
    return result.note.uuid


# --- The LLM's part ---

SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": list(RELATIONS)},
        "insight": {"type": "string"},
        "question": {"type": "string"},
        "note_title": {"type": "string"},
    },
    "required": ["relation", "insight", "question", "note_title"],
}

SYSTEM = (
    "You help someone think with their own notes (podcast transcripts, saved Reddit threads, screenshots). "
    "Given two passages their knowledge graph linked, say plainly what connects them: the shared idea, how one "
    "sheds light on the other, or where they disagree. Be concrete and specific to these passages, never generic. "
    "If they only share words and no idea, say relation 'unrelated'."
)


def prompt(card: dict) -> str:
    def side(s: dict) -> str:
        kind = {"audio": "podcast", "image": "screenshot", "reddit": "Reddit save"}.get(s["source_type"], s["source_type"])
        what = "Passage" if s["passage"] else "Summary"
        return f"{kind}: \"{s['title']}\"{' (' + s['date'] + ')' if s['date'] else ''}\n{what}: {s['text']}"

    shared = f"\nShared names and topics: {', '.join(card['shared'])}" if card.get("shared") else ""
    return (
        f"Note A - {side(card['a'])}\n\nNote B - {side(card['b'])}{shared}\n\n"
        "relation: does B agree with A, contradict it, extend it (adds something new), or apply it (a case of "
        "it in practice)? Or are they unrelated?\n"
        "insight: one or two sentences on what links them, written to the reader (\"you\"), naming the specifics. "
        "Refer to each note by its subject or speaker (\"Ed Maunder's point on durability\"), never as Note A or B "
        "or passage A or B: the reader sees the notes, not those labels.\n"
        "question: one short question worth thinking about, from putting the two together.\n"
        "note_title: a title for a note of the reader's own about this connection (a claim, max 10 words)."
    )


def generate(card: dict) -> dict:
    """The LLM's take on a card: {relation, insight, question, note_title}. Raises LLMError."""
    from .llm import LLMError, chat

    answer = chat(prompt(card), schema=SCHEMA, system=SYSTEM, temperature=0.4)
    if answer.get("relation") not in RELATIONS or not str(answer.get("insight") or "").strip():
        raise LLMError(f"Unusable answer: {str(answer)[:200]}")
    return {k: str(answer.get(k) or "").strip() for k in SCHEMA["required"]}


def pending(data: Optional[dict] = None) -> List[str]:
    """Cards still without the LLM's take, today's first, then the ones picked ahead."""
    data = data or _load()
    order = list(data["days"].get(_today(), [])) + [k for k in data["cards"] if k not in _scheduled(data)]
    return [k for k in dict.fromkeys(order) if k in data["cards"] and data["cards"][k]["insight"] is None
            and data["cards"][k]["verdict"] is None and data["cards"][k]["attempts"] < MAX_ATTEMPTS]


def pending_count() -> int:
    with _lock:
        return len(pending())


def store(key: str, insight: Optional[dict] = None, error: Optional[str] = None) -> None:
    """The LLM's take on a card (or why it failed: tried again up to MAX_ATTEMPTS)."""
    with _lock:
        data = _load()
        card = data["cards"].get(key)
        _claimed.pop(key, None)
        if card is None:
            return
        if insight is not None:
            card["insight"] = insight
        else:
            card["attempts"] += 1
            card["error"] = (error or "failed")[:300]
        _save(data)


def claim(limit: int, lease_seconds: float) -> List[dict]:
    """Cards for the worker's LLM, held for lease_seconds (a heartbeat renews them)."""
    now = time.monotonic()
    with _lock:
        for key, until in list(_claimed.items()):
            if until < now:
                del _claimed[key]
        data = _load()
        keys = [k for k in pending(data) if k not in _claimed][:limit]
        for key in keys:
            _claimed[key] = now + lease_seconds
        return [data["cards"][k] for k in keys]


def renew(lease_seconds: float) -> None:
    with _lock:
        until = time.monotonic() + lease_seconds
        for key in _claimed:
            _claimed[key] = until


def release(keys: List[str]) -> None:
    with _lock:
        for key in keys:
            _claimed.pop(key, None)


def generate_pending(limit: int = UPCOMING + PER_DAY) -> int:
    """Standalone: prepare waiting cards with the local LLM (one thread at a time). Returns how many."""
    from .llm import LLMError

    if _generating.is_set():
        return 0
    _generating.set()
    done = 0
    try:
        for key in pending()[:limit]:
            from . import runstate

            if runstate.active_run():  # a podcast run needs the GPU: the rest wait for the next visit
                break
            with _lock:
                card = _load()["cards"].get(key)
            if card is None:
                continue
            try:
                store(key, generate(card))
                done += 1
            except LLMError as e:
                logger.warning(f"Insight for {key} failed: {e}")
                store(key, error=str(e))
                break  # the LLM is down or confused: try again on the next visit
    finally:
        _generating.clear()
    return done


def generate_in_background() -> bool:
    """Standalone: start preparing waiting cards unless already at it (or a run holds the GPU)."""
    from . import runstate

    if settings.is_primary or _generating.is_set() or runstate.active_run() or not pending_count():
        return False
    threading.Thread(target=generate_pending, name="insights", daemon=True).start()
    return True
