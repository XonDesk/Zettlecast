"""
Markdown Formatter for Vision Results

Formats vision extraction data into structured markdown for note full_text.
"""

import re

from .models import VisionExtraction

# media type -> what to do with it
MEDIA_VERBS = {"book": "to read", "article": "to read", "paper": "to read", "comic": "to read",
               "film": "to watch", "movie": "to watch", "show": "to watch", "series": "to watch",
               "video": "to watch", "course": "to watch", "documentary": "to watch",
               "podcast": "to listen", "album": "to listen", "song": "to listen", "music": "to listen",
               "game": "to play"}
SAVED_TAGS = {"quote": "quote", "idea": "idea", "tip": "tip", "place": "place to visit",
              "product": "to buy", "recipe": "recipe", "reference": "reference"}
TITLE_PREFIX = {"place": "Place", "product": "To buy", "recipe": "Recipe"}

# screen text lines that are the app, not the content: buttons, prompts, tabs, counts, the status bar
CHROME_WORDS = {
    "reply", "replies", "share", "save", "follow", "following", "subscribe", "vote", "gif", "comments",
    "likes", "thread", "top", "back", "< back", "join", "join the conversation", "send message",
    "add a comment", "add comment", "add a reply", "add your reply", "share your thoughts", "show replies",
    "show more replies", "learn more", "shop now", "view activity >", "messages", "home", "create", "chat",
    "inbox", "reels", "instagram", "youtube", "x", "firefox", "edge", "chrome", "safari", "more",
    "hide replies", "send gift", "posts you may like", "order now", "< reddit", "< back thread", "whatsapp",
}
CHROME_RES = [
    re.compile(r"\d{1,2}:\d{2}( (lte|[345]g\+?|wi-?fi))?( \d{1,3}%?)?"),  # status bar: clock, network, battery
    re.compile(r"[\w-]+(\.[\w-]+)+"),  # a bare domain: the context names the site
    re.compile(r"\d[\d.,]*[km]?"),  # bare counts
    re.compile(r"(reply|replies|like|likes|upvote|share|vote|comments) \d[\d.,]*[km]?"),
    re.compile(r"\d+(:\d\d)+ / \d+(:\d\d)+|[\d ]+"),  # video position, stray numbers
    re.compile(r"(add a comment|add a reply|top \d+% \w+).*"),
    re.compile(r"\d[\d.,]*[km]? (upvotes?|points?|votes?|likes?|comments?|replies|views|shares)"
               r"( [·•] \d[\d.,]*[km]? \w+)*"),
    re.compile(r"((view|show) )?(\d+ )?more repl(y|ies)"),
    re.compile(r"\d+ people typing"),
]
# "CandidAd9050 · 17h", "diceclimber OP · 1h", "@runner • 9d ago", "use.origin Sponsored"
AUTHOR_RE = re.compile(
    r"(?P<name>@?(?=[\w.\-]*[a-z])\w[\w.\-]*)(?P<op> OP)?\s*(?:[·•]\s*)?"
    r"(?P<when>\d+\s?(?:[smhdwy]|mo|min|mins|hr|hrs)(?: ago)?|\d+ (?:minute|hour|day|week|month|year)s? ago"
    r"|just now|sponsored)(?: [·•] author)?",
    re.IGNORECASE)
# a post's time on its own line, under the author
WHEN_RE = re.compile(r"\d+ (minute|hour|day|week|month|year)s? ago|\d+[smhdw]|just now", re.IGNORECASE)
HANDLE_RE = re.compile(r"@?(?=[\w.\-]*[a-z])\w[\w.\-]*", re.IGNORECASE)
LINE_END = tuple('.!?:;"”)]…')


def saved_tag(vision: VisionExtraction) -> str:
    """The tag for what a screenshot was kept as ("to read", "place to visit"...), "" for nothing."""
    if vision.saved_as == "media":
        return MEDIA_VERBS.get(vision.item.get("type", "").lower(), "media")
    return SAVED_TAGS.get(vision.saved_as, "")


def saved_title(vision: VisionExtraction) -> str:
    """'Book: Dune', 'Place: Kamikochi', or the model's title when the item has no name."""
    name = vision.item.get("name", "")
    if vision.saved_as == "media" and name:
        kind = vision.item.get("type", "").strip()
        return f"{kind.capitalize()}: {name}" if kind else name
    if vision.saved_as in TITLE_PREFIX and name:
        return f"{TITLE_PREFIX[vision.saved_as]}: {name}"
    return vision.title


def kept_lines(vision: VisionExtraction) -> list[str]:
    """'## Kept' section: what it was kept as and the thing itself (indexed, so it's searchable)."""
    tag, item = saved_tag(vision), vision.item
    if not tag or not item.get("name"):
        return []
    line = f"**{tag.capitalize()}**: {item['name']}"
    if item.get("type") and vision.saved_as != "media":
        line += f" ({item['type']})"
    if item.get("by"):
        line += f" - {item['by']}"
    lines = ["## Kept", "", line]
    if item.get("why"):
        lines.append(f"- **Why**: {item['why']}")
    if item.get("where"):
        lines.append(f"- **Where**: {item['where']}")
    if item.get("price"):
        lines.append(f"- **Price**: {item['price']}")
    return lines + [""]


def _is_chrome(line: str) -> bool:
    low = line.lower().rstrip(".… ")
    return low in CHROME_WORDS or any(r.fullmatch(low) for r in CHROME_RES)


def _letters(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def screen_text(text: str, focus: str = "") -> str:
    """
    A screenshot's text as readable markdown: one block per post / comment (author in bold,
    "17h" dropped), without the app's buttons, counts and status bar or ads, wrapped lines
    joined, and the focus (shown in its own section) not repeated. Markdown would otherwise
    run all the lines together into one paragraph.
    """
    focus_letters = _letters(focus)
    blocks: list[dict] = []  # {"head": str, "lines": [...], "ad": bool, "had_focus": bool}

    def new_block(head: str = "", ad: bool = False) -> dict:
        blocks.append({"head": head, "lines": [], "ad": ad, "had_focus": False})
        return blocks[-1]

    block = new_block()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if block["lines"]:
                block = new_block()
            continue
        author = AUTHOR_RE.fullmatch(line)
        if author or line.lower() == "sponsored":
            ad = not author or author["when"].lower() == "sponsored"
            head = "" if ad else f"**{author['name']}**" + (" (OP)" if author["op"] else "")
            block = new_block(head, ad)
            continue
        lines = block["lines"]
        if WHEN_RE.fullmatch(line):
            if lines and HANDLE_RE.fullmatch(lines[-1]):
                block = new_block(f"**{lines.pop()}**")
            continue
        if _is_chrome(line):
            continue
        joins = line[0].islower() and not line.startswith("r/") or lines and lines[-1].endswith(",")
        if lines and not lines[-1].endswith(LINE_END) and joins:
            hyphen = lines[-1].endswith("-") and lines[-1][-2:-1].isalpha()
            lines[-1] = lines[-1][:-1] + line if hyphen else f"{lines[-1]} {line}"
        else:
            lines.append(line)

    out = []
    for b in blocks:
        if b["ad"]:
            continue
        # the vision prompt asks for the author alone on a block's first line
        first = b["lines"][0] if len(b["lines"]) > 1 and not b["head"] else ""
        if first.removesuffix(" OP") and HANDLE_RE.fullmatch(first.removesuffix(" OP")):
            b["head"] = f"**{first.removesuffix(' OP')}**" + (" (OP)" if first.endswith(" OP") else "")
            b["lines"] = b["lines"][1:]
        if focus_letters:
            # lines of the focus; a short one ("Ulcers") only next to a longer one
            inside = [bool(_letters(l)) and _letters(l) in focus_letters for l in b["lines"]]
            drop = [ins and len(_letters(l)) >= 15 for l, ins in zip(b["lines"], inside)]
            for i, ins in enumerate(inside):
                if ins and ((i and drop[i - 1]) or (i + 1 < len(drop) and drop[i + 1])):
                    drop[i] = True
            b["had_focus"] = any(drop)
            b["lines"] = [l for l, d in zip(b["lines"], drop) if not d]
        lines = b["lines"] or (["_(the focus, above)_"] if b["had_focus"] and b["head"] else [])
        if lines:
            out.append("  \n".join(([b["head"]] if b["head"] else []) + lines))
    return "\n\n".join(out)


def format_vision_result(
    vision: VisionExtraction,
    width: int,
    height: int,
    file_format: str = "unknown",
) -> str:
    """
    Format vision extraction into searchable markdown.

    A screenshot with a focus (the post / comment it was taken for) keeps everything else on
    the screen in "## Screen Text", which isn't embedded (chunker.UNINDEXED_SECTIONS): the
    other comments and the page around it would otherwise link the note to unrelated topics.
    Without a focus, all the text is indexed. File details are never indexed.
    """
    megapixels = (width * height) / 1_000_000
    aspect_ratio = width / height if height > 0 else 1.0
    text = vision.detected_text.strip()
    has_text = bool(text) and text != "No text detected."
    if has_text and vision.kind == "screenshot":
        text = screen_text(text, vision.focus)
        has_text = bool(text)
    elif has_text:
        text = "  \n".join(text.splitlines())  # signs, labels: keep the lines

    sections = kept_lines(vision) + ["## Description", "", vision.description, ""]
    if vision.focus:
        sections += ["## Focus", "", vision.focus, ""]
    if vision.context:
        sections += ["## Context", "", vision.context, ""]
    if has_text and not vision.focus:
        sections += ["## Text", "", text, ""]

    sections += ["## Concepts", ""]
    sections += [f"- {c}" for c in vision.concepts] or ["_None detected_"]
    sections.append("")

    if has_text and vision.focus:
        sections += ["## Screen Text", "", text, ""]
    sections += [
        "## Image Details",
        "",
        f"- **Kind**: {vision.kind}",
        f"- **Format**: {file_format.upper()}",
        f"- **Dimensions**: {width} × {height} pixels ({megapixels:.2f} MP, {aspect_ratio:.2f}:1)",
    ]
    return "\n".join(sections)
