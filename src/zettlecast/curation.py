"""
Comment curation for Reddit saves: the LLM keeps the comments worth finding again (products,
advice, firsthand experience, insight, sources, corrections) as one-line takeaways and drops the
rest (jokes, banter, debate, agreement, bare opinion).

What counts as worth keeping is the rubric: plain text the user can edit (Settings page,
<storage>/reddit_curation.md), with corrections from reviews added as examples. The takeaways are
indexed; the raw comments stay on the note in an unindexed section.
"""

import logging
from pathlib import Path
from typing import List, Optional

from .config import settings
from .llm import LLMError, chat

logger = logging.getLogger(__name__)

COMMENT_CHARS = 1_500  # per comment in the prompt
POST_CHARS = 3_000

KEEP_KINDS = ("product", "advice", "experience", "insight", "resource", "correction")
DROP_KINDS = ("joke", "banter", "debate", "agreement", "opinion", "question", "off-topic", "other")
KIND_LABELS = {"product": "Product", "advice": "Advice", "experience": "Experience", "insight": "Insight",
               "resource": "Resource", "correction": "Correction"}

DEFAULT_RUBRIC = """I save Reddit threads mostly for comments that share something useful.

Keep a comment when it has:
- a product, piece of gear, tool, app, book or service someone recommends or warns against (say which and why)
- practical advice or how-to steps
- firsthand experience with concrete detail: what they tried, what happened, numbers
- insight: an explanation, data, a study or source, or expertise that adds to the post
- a correction of the post, with the reason

Drop a comment when it is:
- a joke, pun, meme or wordplay, however funny
- banter or conversation, or debate and arguing for its own sake
- agreement or praise with nothing new ("this", "great post")
- an opinion with no reason or experience behind it
- a question with no answer, or off topic

When in doubt, keep it only if I'd want to find that information again months from now."""

PROMPT = """You curate the comments of a Reddit thread the user saved, for their personal knowledge base.

The user's rules for what is worth keeping:
<rules>
{rubric}
</rules>
{saved}
For every numbered comment return:
- "n": its number
- "keep": true only if the rules say to keep it
- "kind": for a kept comment one of {keep_kinds}; for a dropped one, one of {drop_kinds}
- "note": for a kept comment, the takeaway in one self-contained line (max 200 characters) that keeps the
  specifics (product names, numbers, steps) so it reads without the thread; for a dropped one, a few
  words on why

Thread: r/{subreddit} · {title}
{post}
Comments:
{comments}"""

SAVED_NOTE = """
The user saved one specific comment (shown under "Saved comment"): that comment is the reason for the save
and is kept anyway. Keep another comment only if it adds something useful to it or to the post.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "comments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "keep": {"type": "boolean"},
                    "kind": {"type": "string", "enum": list(KEEP_KINDS + DROP_KINDS)},
                    "note": {"type": "string"},
                },
                "required": ["n", "keep", "kind", "note"],
            },
        },
    },
    "required": ["comments"],
}


def rubric_path() -> Path:
    return settings.storage_path / "reddit_curation.md"


def rubric() -> str:
    """The user's rubric, or the default one."""
    try:
        text = rubric_path().read_text(encoding="utf-8").strip()
        if text:
            return text
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.error(f"Curation rubric unreadable, using the default: {e}")
    return DEFAULT_RUBRIC


def save_rubric(text: str) -> None:
    """Store the rubric; empty text goes back to the default."""
    path = rubric_path()
    if not text.strip():
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def curate(title: str, subreddit: str, post: str, comments: List[dict],
           saved: Optional[dict] = None) -> List[dict]:
    """
    One verdict per comment: {n, author, keep, kind, note}, in the comments' order.
    Raises LLMError when the LLM is unavailable or answers with nothing usable.
    """
    if not comments:
        return []
    post_text = f"Post:\n{_cut(post, POST_CHARS)}\n" if post.strip() else ""
    if saved:
        post_text += f"\nSaved comment by u/{saved['author']}:\n{_cut(saved['text'], COMMENT_CHARS)}\n"
    numbered = "\n\n".join(f"[{i}] u/{c['author']}:\n{_cut(c['text'], COMMENT_CHARS)}"
                           for i, c in enumerate(comments, 1))
    result = chat(PROMPT.format(rubric=rubric(), saved=SAVED_NOTE if saved else "",
                                keep_kinds=", ".join(KEEP_KINDS), drop_kinds=", ".join(DROP_KINDS),
                                subreddit=subreddit, title=title, post=post_text, comments=numbered),
                  schema=SCHEMA, temperature=0.1)
    by_n = {v.get("n"): v for v in result.get("comments") or [] if isinstance(v, dict)}
    if not by_n:
        raise LLMError("Curation returned no verdicts")
    verdicts = []
    for i, c in enumerate(comments, 1):
        v = by_n.get(i) or {}
        keep = bool(v.get("keep")) and v.get("kind") in KEEP_KINDS and bool((v.get("note") or "").strip())
        verdicts.append({"n": i, "author": c["author"], "keep": keep, "kind": v.get("kind") or "other",
                         "note": (v.get("note") or "").strip()})
    return verdicts


def section(verdicts: List[dict], heading: str = "From the comments") -> str:
    """The kept takeaways as a markdown section, "" when nothing was kept."""
    lines = [f"- **{KIND_LABELS[v['kind']]}**: {v['note']} (u/{v['author']})" for v in verdicts if v["keep"]]
    return f"## {heading}\n" + "\n".join(lines) if lines else ""
