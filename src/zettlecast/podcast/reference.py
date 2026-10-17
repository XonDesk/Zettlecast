"""
Published transcripts (podcast:transcript) as an independent reference.

- speaker_names_from_reference(): when the publisher named the speakers, map
  our diarization labels to those names by time overlap
- compare(): word and speaker agreement between our note and the published
  transcript, with the passages that differ most (for spot-checking)

Both transcripts are usually machine-made: where they agree both are very
likely right; where they differ at least one is wrong.
"""

import bisect
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx

from .rss import USER_AGENT

logger = logging.getLogger(__name__)

# "Speaker 1", "SPEAKER_00", "Unknown" etc. are not names
_GENERIC_SPEAKER = re.compile(r"^(speaker|spk|unknown|guest|host)?[\s_-]*\d*$", re.I)
_CUE_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d{3})|(\d{2}):(\d{2})[.,](\d{3})")
_VOICE = re.compile(r"<v(?:\.[^ >]*)?\s+([^>]+)>")
_NOTE_LINE = re.compile(r"^\[([\d.]+)s - ([\d.]+)s\] ([^:]+): (.*)$")
FILLERS = {"um", "uh", "umm", "uhh", "mm", "hmm", "mhm", "ah", "er", "erm", "oh"}

# (start, end, speaker, text)
Turn = Tuple[float, float, str, str]


@dataclass
class RefWord:
    start: float
    end: float
    speaker: Optional[str]
    text: str


@dataclass
class DiffSpan:
    time: float
    reference: str
    ours: str


@dataclass
class VerifyReport:
    reference_words: int
    our_words: int
    substitutions: int
    missing: int  # in the reference, not in ours (often removed fillers/repeats)
    extra: int  # in ours, not in the reference
    speaker_agreement: Optional[float]
    speaker_mapping: Dict[str, str] = field(default_factory=dict)
    name_mismatches: List[Tuple[str, str]] = field(default_factory=list)
    top_differences: List[Tuple[str, str, int]] = field(default_factory=list)
    spans: List[DiffSpan] = field(default_factory=list)

    @property
    def disagreement(self) -> float:
        return (self.substitutions + self.missing + self.extra) / max(self.reference_words, 1)


def is_named(speaker: Optional[str]) -> bool:
    return bool(speaker and speaker.strip() and not _GENERIC_SPEAKER.match(speaker.strip()))


def normalize(text: str) -> List[str]:
    """Lowercase words without punctuation or fillers, for comparison."""
    text = text.lower().replace("’", "'").replace("-", " ").replace("—", " ")
    words = (w.strip("'") for w in re.sub(r"[^\w' ]+", " ", text).split())
    return [w for w in words if w and w not in FILLERS]


# --- Loading ---


def cache_path_for(audio_path: Path, url: str) -> Path:
    """Where the published transcript is kept: next to the episode audio."""
    ext = Path(urlparse(url).path).suffix or ".txt"
    return audio_path.with_name(f"{audio_path.stem}.transcript{ext}")


def load_reference(url: str, cache_path: Optional[Path] = None) -> List[RefWord]:
    """Download (or read the cached) published transcript and split it into words."""
    if cache_path and cache_path.exists():
        raw = cache_path.read_text(encoding="utf-8")
    else:
        response = httpx.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60)
        response.raise_for_status()
        raw = response.text
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(raw, encoding="utf-8")
    return parse_reference(raw)


def parse_reference(raw: str) -> List[RefWord]:
    try:
        data = json.loads(raw)
    except ValueError:
        return _parse_cues(raw)
    segments = data.get("segments", []) if isinstance(data, dict) else data
    words: List[RefWord] = []
    for seg in segments:
        words += _split(seg.get("startTime", 0), seg.get("endTime", 0), seg.get("speaker"), seg.get("body", ""))
    return words


def _split(start: float, end: float, speaker: Optional[str], body: str) -> List[RefWord]:
    """A cue or segment -> words with evenly spread times."""
    tokens = body.split()
    if not tokens:
        return []
    step = max(float(end) - float(start), 0.0) / len(tokens)
    return [
        RefWord(float(start) + i * step, float(start) + (i + 1) * step, speaker, tok)
        for i, tok in enumerate(tokens)
    ]


def _seconds(match: re.Match) -> float:
    if match.group(1) is not None:
        h, m, s, ms = (int(match.group(i)) for i in range(1, 5))
    else:
        h, (m, s, ms) = 0, (int(match.group(i)) for i in range(5, 8))
    return h * 3600 + m * 60 + s + ms / 1000


def _parse_cues(raw: str) -> List[RefWord]:
    """WebVTT or SRT; VTT voice tags (<v Name>) give the speaker."""
    words: List[RefWord] = []
    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n")):
        lines = block.strip().split("\n")
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing is None:
            continue
        times = list(_CUE_TIME.finditer(lines[timing]))
        if len(times) < 2:
            continue
        text = " ".join(lines[timing + 1:])
        voice = _VOICE.search(text)
        speaker = voice.group(1).strip() if voice else None
        text = re.sub(r"<[^>]+>", "", text)
        words += _split(_seconds(times[0]), _seconds(times[1]), speaker, text)
    return words


# --- Speaker names ---


def _confusion(turn_starts: Sequence[float], turns: Sequence[Turn], ref: List[RefWord]) -> Counter:
    """(reference speaker, our speaker) -> words, matching each reference word by time."""
    counts: Counter = Counter()
    for w in ref:
        mid = (w.start + w.end) / 2
        i = bisect.bisect_right(turn_starts, mid) - 1
        if i >= 0 and turns[i][0] <= mid <= turns[i][1] + 0.5:
            counts[(w.speaker, turns[i][2])] += 1
    return counts


def _greedy_mapping(counts: Counter) -> Dict[str, str]:
    """One-to-one reference speaker -> our speaker, largest overlaps first."""
    mapping: Dict[str, str] = {}
    used = set()
    for (ref_speaker, ours), _ in counts.most_common():
        if ref_speaker not in mapping and ours not in used:
            mapping[ref_speaker] = ours
            used.add(ours)
    return mapping


def speaker_names_from_reference(
    turns: Sequence[Turn], ref: List[RefWord], min_share: float = 0.6, min_words: int = 20
) -> Dict[str, str]:
    """
    Our speaker label -> the publisher's name for that speaker.

    Only named reference speakers count, and a label is named only when most of
    its overlapping reference words carry that name.
    """
    named = [w for w in ref if is_named(w.speaker)]
    if not named:
        return {}
    turns = sorted(turns)
    counts = _confusion([t[0] for t in turns], turns, named)
    per_label: Counter = Counter()
    for (_, ours), n in counts.items():
        per_label[ours] += n
    names = {}
    for ref_speaker, ours in _greedy_mapping(counts).items():
        n = counts[(ref_speaker, ours)]
        if n >= min_words and n >= min_share * per_label[ours]:
            names[ours] = ref_speaker.strip()
    return names


# --- Comparison ---


def note_turns(note_text: str) -> List[Turn]:
    """Transcript lines of a Zettlecast note."""
    body = note_text.split("## Transcript", 1)[-1]
    turns = []
    for line in body.splitlines():
        m = _NOTE_LINE.match(line.strip())
        if m:
            turns.append((float(m[1]), float(m[2]), m[3].strip(), m[4]))
    return turns


def compare(turns: Sequence[Turn], ref: List[RefWord], max_spans: int = 15) -> VerifyReport:
    import jiwer

    ref_tokens: List[str] = []
    ref_times: List[float] = []
    for w in ref:
        for tok in normalize(w.text):
            ref_tokens.append(tok)
            ref_times.append(w.start)
    hyp_tokens = normalize(" ".join(t[3] for t in turns))
    if not ref_tokens or not hyp_tokens:
        return VerifyReport(len(ref_tokens), len(hyp_tokens), 0, len(ref_tokens), len(hyp_tokens), None)

    out = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
    chunks = out.alignments[0]

    diffs: Counter = Counter()
    for c in chunks:
        if c.type == "substitute":
            diffs[(" ".join(ref_tokens[c.ref_start_idx:c.ref_end_idx]),
                   " ".join(hyp_tokens[c.hyp_start_idx:c.hyp_end_idx]))] += 1

    # Spot-check spans: substitutions plus neighbouring differences within 3 words
    spans: List[Tuple[int, int, int, int, int]] = []  # ref_start, ref_end, hyp_start, hyp_end, weight
    for c in chunks:
        if c.type == "equal":
            continue
        if spans and c.ref_start_idx - spans[-1][1] <= 3:
            rs, _, hs, _, weight = spans[-1]
            spans[-1] = (rs, c.ref_end_idx, hs, c.hyp_end_idx, weight + (c.type == "substitute"))
        else:
            spans.append((c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx,
                          int(c.type == "substitute")))
    top = sorted((s for s in spans if s[4]), key=lambda s: -s[4])[:max_spans]
    diff_spans = [
        DiffSpan(
            time=ref_times[min(rs, len(ref_times) - 1)],
            reference=" ".join(ref_tokens[max(rs - 3, 0):r_end + 3]),
            ours=" ".join(hyp_tokens[max(hs - 3, 0):h_end + 3]),
        )
        for rs, r_end, hs, h_end, _ in sorted(top)
    ]

    turns = sorted(turns)
    counts = _confusion([t[0] for t in turns], turns, ref)
    mapping = _greedy_mapping(counts)
    covered = sum(counts.values())
    agree = sum(n for (r, o), n in counts.items() if mapping.get(r) == o)
    mismatches = [(r, o) for r, o in mapping.items() if is_named(r) and r.strip() != o]

    return VerifyReport(
        reference_words=len(ref_tokens),
        our_words=len(hyp_tokens),
        substitutions=out.substitutions,
        missing=out.deletions,
        extra=out.insertions,
        speaker_agreement=agree / covered if covered else None,
        speaker_mapping=mapping,
        name_mismatches=mismatches,
        top_differences=[(r, h, n) for (r, h), n in diffs.most_common(12)],
        spans=diff_spans,
    )
