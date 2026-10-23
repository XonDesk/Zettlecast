"""
What a note is at a glance, for the graph's hover bar and the daily insight cards: a few
sentences (its summary, image description or start), whether it has a picture, its date.
"""

import re
from typing import Optional


BRIEF_CHARS = 600
# where a note says what it is, by type: transcripts have a Summary, Reddit saves the saved
# comment or the post, screenshots a Description, web pages the article itself
BRIEF_SECTIONS = ("Summary", "Saved comment", "Post", "Description", "Linked article")


def note_brief(custom: dict, full_text: str = "") -> str:
    """
    A few sentences on what a note is (the graph shows it on hover): its LLM summary or image
    description, else the section that says it, else its start. Without full_text, only the first two.
    """
    text = ((custom or {}).get("summary") or (custom or {}).get("description") or "").strip()
    if not text and full_text:
        sections = {}
        for part in re.split(r"(?m)^(?=## )", full_text):
            head, _, body = part.partition("\n")
            if head.startswith("## "):
                sections.setdefault(head[3:].split(":")[0].strip(), body.strip())
        text = next((sections[s] for s in BRIEF_SECTIONS if sections.get(s)), "")
        if not text:
            # no known section: the first paragraphs that aren't headings
            text = "\n\n".join(p.strip() for p in full_text.split("\n\n")
                               if p.strip() and not p.lstrip().startswith("#"))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > BRIEF_CHARS:
        text = text[:BRIEF_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def has_image(source_type: str, custom: dict) -> bool:
    """A note whose picture may be shown: a screenshot kept with its image (not for its information only)."""
    return source_type == "image" and custom.get("image_kept") is not False


def iso_date(value) -> Optional[str]:
    """YYYY-MM-DD from an RSS date ("Wed, 27 Nov 2019 ...") or an ISO one."""
    if not value:
        return None
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(str(value)).date().isoformat()
    except (TypeError, ValueError):
        return str(value)[:10] if re.match(r"\d{4}-\d{2}-\d{2}", str(value)) else None
