"""
Automatic transcript verification: a correction is applied only if the audio supports it.

Proposers suggest corrections; the acoustic judge (podcast/acoustic.py) scores
each one against the audio of that passage; support from independent sources
breaks ties between candidates that fit the audio about equally well.

Proposers:
- the LLM "suspects" pass: likely misheard names/terms with sound-alike candidates
- the publisher's transcript (podcast:transcript), where it heard something else
- the show glossary: spellings written down outside the pipeline in earlier episodes
  (publisher's speaker labels, show notes, Wikipedia pages for LLM-proposed names)
- an online lookup (podcast/lookup.py) for names nothing else confirms: real
  names found on Wikipedia through what the passage says about them confirm a
  spelling, and a close result-page match is proposed
- the LLM cleanup pass: every content edit it makes is checked the same way

Rules (penalty = extra negative log-likelihood per changed word):
- a candidate over ACCEPT_PER_WORD doesn't fit the audio and is never applied
- among plausible candidates the best supported one wins if it has more support
  than what was recognised: LLM confidence (high 2, medium 1, low 0.5),
  publisher agreement (+1), glossary/known spelling (+1.5), a real name found
  by the lookup (+2.5, enough to beat a confident LLM guess; the heard name gets
  it too when it is real); the recognised
  text starts at 1 (+1 if the publisher heard it too)
- the winner must also be backed by evidence for what it changes:
  - a respelling of a heard name (same sounds, other letters: "Coop Cast" -> "KoopCast"; the
    recogniser wrote a capitalised word that is no ordinary word of this transcript) changes
    only the spelling, which the LLM and both recognisers often get wrong for rare names, so
    every capitalised word of the new text must have a confirmed spelling (vocabulary,
    glossary, found online) that the old text lacks
  - other changes ("all So" -> "Jason") change what was said: proposed by the LLM, the
    glossary or the lookup, and heard by the publisher or preferred by the audio (penalty <= 0)
  candidates that fit but aren't backed go to Needs Review when they look like names
- cleanup edits are checked the same way (deletions of fillers always pass)
"""

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .enhancer import _key, _words
from .models import TranscriptSegment
from .reference import normalize

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
ACCEPT_PER_WORD = 10.0
CONTEXT_SECONDS = 3.0
PAD_SECONDS = 0.15
MAX_SPAN_TOKENS = 6
# Glossary evidence per episode: the publisher's speaker label, or an LLM-proposed name a
# Wikipedia page confirms, is trusted at once; show notes and titles have their own
# misspellings ("Batiste Moral" next to "Baptiste Morel-Prieur"), so a name written only
# there is trusted once a second episode confirms it
SOURCE_WEIGHT = {"speaker": 2, "wikipedia": 2, "show notes": 1}
TRUSTED_COUNT = 2
CONFIDENCE_SUPPORT = {"high": 2.0, "medium": 1.0, "low": 0.5}
# The LLM is often confidently wrong about rare names ("Christian Blomfeldt", high),
# so a real name found online must outweigh its top confidence
LOOKUP_SUPPORT = 2.5
MAX_LOOKUPS = 25  # per episode
# Parakeet prices rare spellings as well as sounds: "Ryne" (said "Ryan") costs 12-25 more
# than "Ryan". A confirmed multi-word name spelled almost like what was heard gets this
# looser limit; different words still cost more ("Ingebrigtsen" for "Eden" ~25)
KNOWN_NAME_ACCEPT = 30.0
KNOWN_NAME_SPELLING = 0.85
# The publisher's transcript can't tell "Ryan" from "Ryne" either, so a respelling of a
# confirmed full name gets enough extra support to outweigh the publisher agreeing with the original
RESPELLING_SUPPORT = 1.5
# Candidate words this close to a known spelling are also tried in that spelling ("Bennett" -> "Benat")
SNAP_CUTOFF = 0.65
# A respelling keeps the sounds: letters mostly agree (ratio) and the length barely changes,
# so dropping or adding a word ("Coach Jason Koop" -> "Jason Koop") is not one
RESPELL_SIMILARITY = 0.6
RESPELL_LENGTH = 0.75
# The audio "prefers" a change when it fits at least as well as what was recognised
AUDIO_PREFERS = 0.0
_PREFIX = re.compile(r"^(\[[\d.]+s - [\d.]+s\] [^:]+: )(.*)$")
_EDGE_PUNCT = re.compile(r"^[^\w'\[]+|[^\w'\]?]+$")


@dataclass
class Unit:
    """A run of words with its time span ("" once deleted)."""

    text: str
    start: float
    end: float


@dataclass
class Correction:
    time: float
    original: str
    chosen: str
    source: str
    penalty: float
    confidence: str
    alternatives: List[str] = field(default_factory=list)
    publisher: bool = False  # the publisher's transcript has the chosen text
    looked_up: bool = False  # the chosen name was found online

    def as_dict(self) -> dict:
        return {
            "time": round(self.time, 1), "original": self.original, "chosen": self.chosen,
            "source": self.source, "penalty": round(self.penalty, 1), "confidence": self.confidence,
            "alternatives": self.alternatives,
        }


@dataclass
class Suspect:
    line: int
    text: str
    candidates: List[str]
    confidence: str = "medium"
    about: str = ""  # who/what it is, for the online lookup ("Norwegian triathlete")
    token: Optional[int] = None  # where in the line, when the text occurs more than once


class Line:
    """One transcript line: fixed "[t] speaker:" prefix plus timed units of text."""

    def __init__(self, prefix: str, units: List[Unit]):
        self.prefix = prefix
        self.units = units

    @property
    def start(self) -> float:
        return self.units[0].start if self.units else 0.0

    @property
    def end(self) -> float:
        return self.units[-1].end if self.units else 0.0

    def content(self) -> str:
        return " ".join(u.text for u in self.units if u.text)

    def text(self) -> str:
        return self.prefix + self.content()

    def tokens(self) -> List[Tuple[str, int]]:
        """(token, unit index) for every word currently in the line."""
        return [(tok, i) for i, u in enumerate(self.units) for tok in u.text.split()]

    def token_times(self, toks: List[Tuple[str, int]], t1: int, t2: int) -> Tuple[float, float]:
        """Time span covered by tokens [t1, t2) (for an insertion, the unit at t1)."""
        if not toks:
            return self.start, self.end
        if t1 >= t2:
            i = min(t1, len(toks) - 1)
            u = self.units[toks[i][1]]
            return u.start, u.end
        return self.units[toks[t1][1]].start, self.units[toks[t2 - 1][1]].end

    def replace(self, t1: int, t2: int, new: List[str]) -> None:
        """Replace tokens [t1, t2) with new tokens, merging the units they belong to."""
        toks = self.tokens()
        if not toks:
            return
        if t1 < t2:
            u1, u2 = toks[t1][1], toks[t2 - 1][1]
        else:
            u1 = u2 = toks[min(t1, len(toks) - 1)][1]
        before = [tok for k, (tok, ui) in enumerate(toks) if ui == u1 and k < t1]
        after = [tok for k, (tok, ui) in enumerate(toks) if ui == u2 and k >= max(t2, t1)]
        if t1 >= len(toks):  # append after the last unit
            before, after = [tok for tok, ui in toks if ui == u1], []
        merged = Unit(" ".join(before + new + after), self.units[u1].start, self.units[u2].end)
        self.units[u1:u2 + 1] = [merged]


def build_lines(segments: Sequence[TranscriptSegment]) -> List[Line]:
    lines = []
    for seg in segments:
        prefix = f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker + ': ' if seg.speaker else ''}"
        if seg.words:
            units = [Unit(w.text, w.start, w.end) for w in seg.words]
        else:
            units = [Unit(seg.text, seg.start, seg.end)]
        lines.append(Line(prefix, units))
    return lines


def _clean_candidate(tokens: List[str], original: List[str]) -> List[str]:
    """Strip a candidate's edge punctuation and give it the original's trailing punctuation."""
    out = [t for t in (_EDGE_PUNCT.sub("", t) for t in tokens) if t]
    trail = re.search(r"[^\w']+$", original[-1]) if original else None
    if out and trail:
        out[-1] += trail.group(0)
    return out


def _changed_words(old: List[str], new: List[str]) -> int:
    """
    New words a candidate brings in. Unchanged words don't dilute a penalty, and
    dropping spoken words earns no allowance ("Ring's group" -> "Ring" is 1 word).
    """
    matcher = difflib.SequenceMatcher(a=[_key(t) for t in old], b=[_key(t) for t in new], autojunk=False)
    return max(1, sum(j2 - j1 for op, i1, i2, j1, j2 in matcher.get_opcodes() if op != "equal"))


def _name_runs(text: str) -> List[str]:
    """
    Capitalised runs of a text ("Cocodona 250" keeps its number; "[[Benat??]]" markers and
    edge punctuation are dropped; a URL or other odd token ends a run).
    """
    runs, run = [], []
    for t in text.split():
        w = _EDGE_PUNCT.sub("", t).strip("[]?")
        if re.fullmatch(r"[\w'’-]+", w) and (w[:1].isupper() or (run and w.isdigit())):
            run.append(w)
            continue
        if run:
            runs.append(" ".join(run))
        run = []
    if run:
        runs.append(" ".join(run))
    return runs


def _name_part(text: str) -> str:
    """The longest capitalised run of a phrase: "Stephen Seiler has" -> "Stephen Seiler"."""
    return max(_name_runs(text), key=lambda r: len(r.split()), default="")


def _has(term: str, text: str) -> bool:
    return re.search(rf"(?<![\w'’-]){re.escape(term)}(?![\w'’-])", text) is not None


def _spoken_names(written: str, transcript: str) -> List[str]:
    """
    Names from human-written text (show notes, title) that the transcript also has: the
    longest piece of each capitalised run found there ("The Cocodona 250" -> "Cocodona 250").
    """
    out = []
    for run in _name_runs(written):
        words = run.split()
        taken = [False] * len(words)
        for n in range(len(words), 0, -1):
            for i in range(len(words) - n + 1):
                term = " ".join(words[i:i + n])
                if any(taken[i:i + n]) or not term[:1].isupper() or len(term) < 3:
                    continue
                if _has(term, transcript):
                    out.append(term)
                    taken[i:i + n] = [True] * n
    return out


def _respells(old: str, new: str) -> bool:
    """Same sounds, other letters ("Coop Cast" -> "KoopCast", "Benette" -> "Bennett")."""
    a, b = "".join(map(_key, old.split())), "".join(map(_key, new.split()))
    if not a or not b or a == b:
        return False
    return (
        min(len(a), len(b)) / max(len(a), len(b)) >= RESPELL_LENGTH
        and difflib.SequenceMatcher(None, a, b).ratio() >= RESPELL_SIMILARITY
    )


def _contains(words: List[str], text: str) -> bool:
    """
    Whether a run of words says text: compounds aside ("ultra marathon" says "ultramarathon"),
    and a name only by its sound, since the publisher can't spell rare names either ("ryan"
    says "Ryne").
    """
    target = "".join(normalize(text))
    if not target:
        return False
    said = [(normalize(w), w[:1].isupper()) for w in text.split()]
    said = [(k, name) for parts, name in said for k in parts]
    for i in range(len(words) - len(said) + 1):
        if all(w == k or (name and _respells(w, k)) for w, (k, name) in zip(words[i:i + len(said)], said)):
            return True
    for i in range(len(words)):
        acc = ""
        for w in words[i:]:
            acc += w
            if acc == target:
                return True
            if len(acc) >= len(target):
                break
    return False


def _caps(text: str) -> Tuple[str, ...]:
    return tuple(_key(w) for w in text.split() if w[:1].isupper() and _key(w))


def _looks_like_name(text: str) -> bool:
    return any(w[:1].isupper() for w in text.split()[1:]) or (text[:1].isupper() and len(text.split()) == 1)


def load_audio(path: Path) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(path)).set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
    return np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0


# --- Glossary ---
#
# A show's glossary holds spellings someone outside the pipeline wrote down: the publisher's
# speaker labels, the episode's show notes or title, or a Wikipedia page for a name the LLM
# proposed. The recogniser, the LLM and the audio judge can tell what was said but not how a
# rare name is spelled, and learning from their own output would feed back into itself.
# Evidence is kept per episode (term -> {episode id: source}), so reprocessing an episode
# replaces its evidence instead of adding to it.


def glossary_path(storage: Path, show: str) -> Path:
    safe = re.sub(r"[^\w\s-]", "", show).strip() or "default"
    return storage / "glossaries" / f"{safe}.json"


def glossary_count(evidence: Dict[str, str]) -> int:
    """Evidence for a term: one source per episode, weighted by how reliable it is."""
    return sum(SOURCE_WEIGHT.get(source, 0) for source in evidence.values())


def load_glossary(path: Path) -> Dict[str, Dict[str, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    # older files held bare counts learned from the pipeline's own output: not evidence
    return {t: dict(ev) for t, ev in data.items() if isinstance(ev, dict) and ev}


def save_glossary(path: Path, glossary: Dict[str, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = dict(sorted(((t, ev) for t, ev in glossary.items() if ev), key=lambda kv: (-glossary_count(kv[1]), kv[0])))
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")


# --- Verifier ---


class TranscriptVerifier:
    """Holds the timed transcript, checks proposed corrections against the audio, applies winners."""

    def __init__(
        self,
        segments: Sequence[TranscriptSegment],
        audio: np.ndarray,
        scorer,
        reference: Optional[list] = None,
        glossary: Optional[Dict[str, Dict[str, str]]] = None,
        accept_per_word: float = ACCEPT_PER_WORD,
        lookup=None,
    ):
        self.lines = build_lines(segments)
        self.audio = audio
        self.scorer = scorer
        self.reference = sorted(reference or [], key=lambda w: w.start)
        self._ref_starts = [w.start for w in self.reference]
        self.glossary = {t: dict(ev) for t, ev in (glossary or {}).items()}
        self.accept = accept_per_word
        self.lookup = lookup  # podcast.lookup.NameLookup or None (offline)
        self.lookups = 0
        self.known: set = set()  # keys of trusted name spellings (set by resolve)
        self.ordinary: set = set()  # words this transcript writes in lowercase (set by resolve/check_cleanup)
        self.rejected: List[dict] = []
        self.accepted_edits = 0
        self.corrections: List[Correction] = []

    def text(self) -> str:
        return "\n".join(line.text() for line in self.lines)

    # --- Acoustic scoring ---

    def penalties(self, line: Line, t1: int, t2: int, candidates: List[List[str]]) -> List[float]:
        """Per-changed-word acoustic penalty of each candidate replacing tokens [t1, t2)."""
        toks = line.tokens()
        if not toks:
            return [float("inf")] * len(candidates)
        words = [t for t, _ in toks]

        def start_of(i: int) -> float:
            return line.units[toks[min(i, len(toks) - 1)][1]].start

        def end_of(i: int) -> float:
            return line.units[toks[min(i, len(toks) - 1)][1]].end

        span_start, span_end = line.token_times(toks, t1, t2)
        w1, w2 = t1, max(t2, t1)
        while w1 > 0 and span_start - start_of(w1 - 1) < CONTEXT_SECONDS:
            w1 -= 1
        while w2 < len(toks) and end_of(w2) - span_end < CONTEXT_SECONDS:
            w2 += 1
        clip_start = min(start_of(w1), span_start)
        clip_end = max(end_of(w2 - 1) if w2 > 0 else span_end, span_end)
        a = max(0, int((clip_start - PAD_SECONDS) * SAMPLE_RATE))
        b = min(len(self.audio), int((clip_end + PAD_SECONDS) * SAMPLE_RATE))
        if b - a < int(0.3 * SAMPLE_RATE):
            return [float("inf")] * len(candidates)
        original = " ".join(words[w1:w2])
        texts = [" ".join(words[w1:t1] + c + words[t2:w2]) for c in candidates]
        scores = self.scorer.nll(self.audio[a:b], [original] + texts)
        changed = [_changed_words(words[t1:t2], c) for c in candidates]
        return [(s - scores[0]) / n for s, n in zip(scores[1:], changed)]

    def _ordinary_words(self) -> set:
        return {w for line in self.lines for w in re.findall(r"\b[a-z][\w']*", line.content())}

    def _backed(self, original: str, new: str, penalty: float, proposed: bool, publisher: bool,
                online: frozenset = frozenset()) -> bool:
        """
        Whether a change that fits the audio has evidence for what it changes (module rules):
        a respelling needs confirmed spellings the original lacks; other changes need a
        proposer plus the publisher hearing them or the audio preferring them.
        """
        # the recogniser heard a name here ("Gustav In"), not ordinary words ("all So")
        heard_name = any(w[:1].isupper() and _key(w) not in self.ordinary for w in original.split())
        # ... and the change respells it ("Jim loses" -> "Jim lost" leaves the name alone)
        if heard_name and _respells(original, new) and _caps(new) and _caps(new) != _caps(original):
            def confirmed(c: Tuple[str, ...]) -> bool:
                return bool(c) and (all(k in self.known for k in c) or c in online)

            return confirmed(_caps(new)) and not confirmed(_caps(original))
        return proposed and (publisher or penalty <= AUDIO_PREFERS)

    # --- Publisher transcript ---

    def _reference_between(self, start: float, end: float, pad: float = 0.3) -> List[str]:
        """Publisher's words whose midpoint falls in [start - pad, end + pad]; punctuation stays attached."""
        i = max(0, np.searchsorted(self._ref_starts, start - pad) - 1) if self.reference else 0
        out: List[str] = []
        for w in self.reference[i:]:
            mid = (w.start + w.end) / 2
            if mid > end + pad:
                break
            text = w.text.strip()
            if mid < start - pad or not text:
                continue
            if not re.search(r"\w", text):
                if out:
                    out[-1] += text
                continue
            out.append(text)
        return out

    def publisher_hints(self) -> List[Tuple[int, str, str]]:
        """(line, ours, theirs) where the publisher heard a content word differently."""
        hints = []
        if not self.reference:
            return hints
        for li, line in enumerate(self.lines):
            ours = [t for t, _ in line.tokens()]
            theirs = [t for t in self._reference_between(line.start, line.end) if t.strip()]
            matcher = difflib.SequenceMatcher(a=[_key(t) for t in ours], b=[_key(t) for t in theirs], autojunk=False)
            for op, i1, i2, j1, j2 in matcher.get_opcodes():
                if op != "replace" or i2 - i1 > 4 or j2 - j1 > 4:
                    continue
                o, t = " ".join(ours[i1:i2]), " ".join(theirs[j1:j2])
                if any(w[:1].isupper() or len(_key(w)) >= 5 for w in (ours[i1:i2] + theirs[j1:j2])):
                    hints.append((li, o, t))
        return hints

    # --- Suspects (LLM / glossary proposals) ---

    def trusted_terms(self) -> List[str]:
        """Glossary terms with enough evidence (count >= TRUSTED_COUNT), most-confirmed first."""
        counts = {t: glossary_count(ev) for t, ev in self.glossary.items()}
        return [t for t in sorted(counts, key=lambda t: -counts[t]) if counts[t] >= TRUSTED_COUNT]

    def glossary_suspects(self) -> List[Suspect]:
        """Capitalised 1-3 word phrases close to, but not exactly, a glossary term."""
        terms = self.trusted_terms()
        if not terms:
            return []
        lowered = {t.lower(): t for t in terms}
        out = []
        for li, line in enumerate(self.lines):
            words = [t for t, _ in line.tokens()]
            for n in (3, 2, 1):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(_EDGE_PUNCT.sub("", w) for w in words[i:i + n])
                    if not phrase[:1].isupper() or phrase.lower() in lowered or len(phrase) < 4:
                        continue
                    match = difflib.get_close_matches(phrase.lower(), list(lowered), n=1, cutoff=0.75)
                    if match and len(match[0].split()) == n:
                        out.append(Suspect(li, phrase, [lowered[match[0]]], "low", token=i))
        return out

    def _find(self, line: Line, text: str, token: Optional[int] = None) -> Optional[Tuple[int, int]]:
        target = [_key(t) for t in text.split() if _key(t)]
        keys = [_key(t) for t, _ in line.tokens()]
        if not target:
            return None
        if token is not None:
            return (token, token + len(target)) if keys[token:token + len(target)] == target else None
        for i in range(len(keys) - len(target) + 1):
            if keys[i:i + len(target)] == target:
                return i, i + len(target)
        return None

    def _look_up(self, suspect: Suspect, known: set) -> Tuple[List[str], List[str]]:
        """
        (full replacements proposed online, real names found) for a suspect's name when
        nothing local confirms it: not a known spelling and no candidate is known either.
        """
        if self.lookup is None or self.lookups >= MAX_LOOKUPS or not suspect.about.strip():
            return [], []
        heard = _name_part(suspect.text)
        if not heard or heard not in suspect.text:
            return [], []
        for text in [heard, *suspect.candidates]:
            words = [_key(w) for w in text.split() if w[:1].isupper() and _key(w)]
            if words and all(k in known for k in words):
                return [], []
        self.lookups += 1
        proposals, real = self.lookup.find(heard, suspect.about)
        return [suspect.text.replace(heard, name, 1) for name in proposals], [*real, *proposals]

    def resolve(self, suspects: List[Suspect], vocabulary: str = "") -> List[Correction]:
        """Score every suspect's candidates against the audio and apply the winners."""
        self.ordinary = self._ordinary_words()
        # Spellings of names we trust: capitalised words of the vocabulary + glossary terms
        known = {_key(w) for w in re.findall(r"[\w'’-]+", vocabulary) if w[:1].isupper()}
        known |= {k for term in self.trusted_terms() for k in _words(term)}
        self.known = known
        # Known spellings with their case, for snapping "Ryan" -> "Ryne" when "Ryne Anderson" is known
        known_words = {
            _key(w): w for w in re.findall(r"[\w'’-]+", vocabulary + " " + " ".join(self.trusted_terms()))
            if w[:1].isupper() and len(w) > 2
        }

        def snapped(text: str) -> str:
            out = []
            for w in text.split():
                core = _EDGE_PUNCT.sub("", w)
                if core[:1].isupper() and _key(core) not in known_words:
                    match = difflib.get_close_matches(_key(core), list(known_words), n=1, cutoff=SNAP_CUTOFF)
                    if match:
                        w = w.replace(core, known_words[match[0]])
                out.append(w)
            return " ".join(out)

        caps = _caps

        # group by (line, span); merge candidates from several proposers
        spans: Dict[Tuple[int, int, int], Dict[str, Tuple[str, float]]] = {}
        found_online: Dict[Tuple[int, int, int], set] = {}  # span -> names (caps keys) the lookup found
        for s in suspects + self.glossary_suspects():
            if not (0 <= s.line < len(self.lines)):
                continue
            found = self._find(self.lines[s.line], s.text, s.token)
            if not found or found[1] - found[0] > MAX_SPAN_TOKENS:
                continue
            slot = spans.setdefault((s.line, *found), {})
            for cand in s.candidates:
                cand = cand.strip()
                if not cand or [_key(w) for w in cand.split()] == [_key(w) for w in s.text.split()]:
                    continue
                source = "glossary" if s.confidence == "low" and cand in self.glossary else "llm"
                support = CONFIDENCE_SUPPORT.get(s.confidence, 1.0) if source == "llm" else 0.0
                for variant in dict.fromkeys([cand, snapped(cand)]):
                    prev = slot.get(variant)
                    slot[variant] = (source, max(support, prev[1] if prev else 0.0))
            proposals, real = self._look_up(s, known)
            if real:
                found_online.setdefault((s.line, *found), set()).update(caps(name) for name in real)
            for full in proposals:
                slot.setdefault(full, ("wikipedia", 0.0))

        # publisher's version of each span joins the candidates
        corrections: List[Correction] = []
        for (li, t1, t2) in sorted(spans, key=lambda k: (k[0], -k[1])):
            line = self.lines[li]
            toks = line.tokens()
            original_tokens = [t for t, _ in toks[t1:t2]]
            original = " ".join(original_tokens)
            start, end = line.token_times(toks, t1, t2)
            theirs = " ".join(self._reference_between(start, end, pad=0.05))
            theirs_key = [_key(w) for w in theirs.split() if _key(w)]
            cands = spans[(li, t1, t2)]
            # word times differ by a few hundredths between the two recognisers, so look a
            # little wider than the span and ask whether the publisher said the candidate there
            around = normalize(" ".join(self._reference_between(start, end)))
            heard_original = _contains(around, original)

            def agrees(text: str) -> bool:
                return not heard_original and _contains(around, text)

            if theirs and theirs_key != [_key(t) for t in original_tokens] and len(theirs_key) <= MAX_SPAN_TOKENS:
                src, sup = cands.get(theirs, ("publisher", 0.0))
                cands[theirs] = (src, sup)
            if not cands:
                continue

            names = list(cands)
            cleaned = [_clean_candidate(c.split(), original_tokens) for c in names]
            penalties = self.penalties(line, t1, t2, cleaned)
            online = found_online.get((li, t1, t2), set())

            def support(text: str, base: float, is_original: bool = False) -> float:
                s = base
                if heard_original if is_original else agrees(text):
                    s += 1.0
                capitalised = caps(text)
                if capitalised and all(k in known for k in capitalised):
                    s += 1.5
                if capitalised and capitalised in online:
                    s += LOOKUP_SUPPORT
                return s

            original_support = support(original, 1.0, is_original=True)  # what was recognised starts at 1
            # (support, tie-break on fit to the audio, ...)
            def respelling(text: str) -> bool:
                """A known multi-word name spelled almost like the original ("Ryan Anderson" -> "Ryne")."""
                capitalised = caps(text)
                return (
                    len(capitalised) >= 2 and all(k in known for k in capitalised)
                    and difflib.SequenceMatcher(None, _key(original), _key(text)).ratio() >= KNOWN_NAME_SPELLING
                )

            options = [
                (support(name, cands[name][1]) + (RESPELLING_SUPPORT if respelling(name) else 0.0),
                 -pen, name, cands[name][0], pen, clean)
                for name, pen, clean in zip(names, penalties, cleaned)
                if pen <= self.accept or (pen <= KNOWN_NAME_ACCEPT and respelling(name))
            ]
            if not options:
                continue


            ranked = sorted(options, reverse=True)
            better = [o for o in ranked if o[0] > original_support]
            backed = [o for o in better if self._backed(
                original, o[2], o[4], proposed=o[3] != "publisher", publisher=agrees(o[2]), online=online,
            )]
            score, _, name, source, pen, clean = (backed or ranked)[0]
            alternatives = [o[2] for o in ranked if o[2] != name]
            if backed:
                confidence = "high" if score - original_support >= 1.0 else "medium"
                line.replace(t1, t2, clean)
                agreed = agrees(" ".join(clean))
                # online evidence counts only when it is for a different name than was heard
                looked_up = bool(caps(name)) and caps(name) in online and caps(name) != caps(original)
                if looked_up and source != "wikipedia":
                    source += "+wikipedia"
                corrections.append(Correction(
                    start, original, " ".join(clean), source, pen, confidence, alternatives, agreed, looked_up
                ))
            elif (better or score >= original_support - 0.25) and (_looks_like_name(original) or _looks_like_name(name)):
                corrections.append(Correction(start, original, original, "kept", 0.0, "low", [name] + alternatives))
        corrections.sort(key=lambda c: c.time)
        self.corrections = corrections
        applied = [c for c in corrections if c.confidence != "low"]
        logger.info(f"Verification: {len(applied)} corrections applied, "
                    f"{len(corrections) - len(applied)} left for review, {len(spans)} spans checked, "
                    f"{self.lookups} names looked up")
        return corrections

    # --- Cleanup edits ---

    def check_cleanup(self, line_offset: int, original_chunk: str, cleaned_chunk: str) -> Tuple[str, List[dict]]:
        """
        Apply the cleanup's edits to lines [line_offset, ...) where the audio supports them.
        Returns the resulting chunk text and the rejected edits.
        """
        rejected = []
        self.ordinary = self._ordinary_words()
        cleaned_lines = cleaned_chunk.split("\n")
        for k, cleaned_line in enumerate(cleaned_lines):
            li = line_offset + k
            if li >= len(self.lines):
                break
            line = self.lines[li]
            m = _PREFIX.match(cleaned_line)
            if not m or m.group(1) != line.prefix:
                continue
            ours = [t for t, _ in line.tokens()]
            new = m.group(2).split()
            matcher = difflib.SequenceMatcher(a=[_key(t) for t in ours], b=[_key(t) for t in new], autojunk=False)
            ops = [op for op in matcher.get_opcodes() if op[0] != "equal"]
            to_score, accepted = [], []
            for op, i1, i2, j1, j2 in ops:
                same_letters = "".join(map(_key, ours[i1:i2])) == "".join(map(_key, new[j1:j2]))
                new_keys = {_key(t) for t in new[j1:j2]}
                drops_known_name = any(
                    t[:1].isupper() and _key(t) in self.known and _key(t) not in new_keys for t in ours[i1:i2]
                )
                if drops_known_name and op != "delete":
                    # never respell a confirmed name ("Ryne" -> "Ryan" fits the audio better)
                    rejected.append({
                        "time": round(line.token_times(line.tokens(), i1, i2)[0], 1),
                        "original": " ".join(ours[i1:i2]), "suggested": " ".join(new[j1:j2]), "penalty": 0.0,
                    })
                    continue
                if op == "delete" or same_letters:
                    accepted.append((i1, i2, new[j1:j2]))
                else:
                    to_score.append((i1, i2, new[j1:j2]))
            for i1, i2, repl in to_score:
                pen = self.penalties(line, i1, i2, [repl])[0]
                start, end = line.token_times(line.tokens(), i1, i2)
                # the publisher said the new words there (fillers and hyphens aside), not the old ones
                around = normalize(" ".join(self._reference_between(start, end)))
                old_text = " ".join(ours[i1:i2])
                heard = _contains(around, " ".join(repl)) and not (old_text and _contains(around, old_text))
                if pen <= self.accept and self._backed(old_text, " ".join(repl), pen, True, heard):
                    accepted.append((i1, i2, repl))
                else:
                    rejected.append({
                        "time": round(line.token_times(line.tokens(), i1, i2)[0], 1),
                        "original": " ".join(ours[i1:i2]), "suggested": " ".join(repl), "penalty": round(pen, 1),
                    })
            # case/punctuation-only differences inside "equal" runs: take the cleanup's
            # version word by word (keeps each word's own timing)
            for op, i1, i2, j1, j2 in matcher.get_opcodes():
                if op == "equal":
                    accepted += [
                        (i1 + k, i1 + k + 1, [new[j1 + k]])
                        for k in range(i2 - i1) if ours[i1 + k] != new[j1 + k]
                    ]
            for i1, i2, repl in sorted(accepted, key=lambda e: (e[0], e[1]), reverse=True):
                line.replace(i1, i2, repl)
            self.accepted_edits += len(accepted)
        self.rejected.extend(rejected)
        n = len(cleaned_lines)
        return "\n".join(line.text() for line in self.lines[line_offset:line_offset + n]), rejected

    def learn(
        self, episode_id: str, speaker_names: Sequence[str] = (), verified: Sequence[str] = (), written: str = "",
    ) -> Dict[str, Dict[str, str]]:
        """
        Record this episode's evidence in the glossary (returned; caller saves it), replacing
        what an earlier run of the same episode recorded. Learned are only spellings written
        down outside the pipeline:
        - verified: speaker names from the publisher's transcript labels
        - written: the show notes and title; names there that the transcript also has
        - corrections the LLM proposed whose name the lookup found online
        LLM-only names, the publisher's transcript text (also speech recognition) and the
        glossary's own proposals are never learned. A lone word counts only where the
        transcript capitalises it mid-sentence, and the words of a full name learned from
        the same episode aren't stored again on their own.
        """
        for ev in self.glossary.values():
            ev.pop(episode_id, None)
        transcript = "\n".join(line.content() for line in self.lines)
        lowercase_words = set(re.findall(r"\b[a-z][\w']*", transcript))
        # capitalised words not at a sentence start: names, not "So" or "Buy"
        mid_sentence = {
            re.sub(r"['’]s$", "", w)
            for line in self.lines for w in re.findall(r"(?<=[\w,;)] )[A-Z][\w'’-]*", line.content())
        }
        terms: Dict[str, str] = {}

        def add(term: str, source: str) -> None:
            term = re.sub(r"['’]s$", "", term.strip())
            if not term:
                return
            if " " not in term and (term.lower() in lowercase_words or term not in mid_sentence):
                return
            if SOURCE_WEIGHT[source] > SOURCE_WEIGHT.get(terms.get(term, ""), 0):
                terms[term] = source

        for name in verified:
            if " " in name.strip():
                add(name, "speaker")
        for c in self.corrections:
            # found online AND proposed by the LLM: two independent sources
            if c.confidence != "low" and c.looked_up and c.source != "wikipedia":
                add(_name_part(c.chosen), "wikipedia")
        for term in _spoken_names(written, transcript + "\n" + "\n".join(speaker_names)):
            add(term, "show notes")
        in_full_names = {w for t in terms if " " in t for w in t.split()}
        for term, source in terms.items():
            if " " in term or term not in in_full_names:
                self.glossary.setdefault(term, {})[episode_id] = source
        self.glossary = {t: ev for t, ev in self.glossary.items() if ev}
        return self.glossary
