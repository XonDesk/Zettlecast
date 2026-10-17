"""
Automatic transcript verification: a correction is applied only if the audio supports it.

Proposers suggest corrections; the acoustic judge (podcast/acoustic.py) scores
each one against the audio of that passage; support from independent sources
breaks ties between candidates that fit the audio about equally well.

Proposers:
- the LLM "suspects" pass: likely misheard names/terms with sound-alike candidates
- the publisher's transcript (podcast:transcript), where it heard something else
- the show glossary: names and terms confirmed in earlier episodes
- the LLM cleanup pass: every content edit it makes is checked the same way

Rules (penalty = extra negative log-likelihood per changed word):
- a candidate over ACCEPT_PER_WORD doesn't fit the audio and is never applied
- among plausible candidates the best supported one wins if it has more support
  than what was recognised: LLM confidence (high 2, medium 1, low 0.5),
  publisher agreement (+1), glossary/known spelling (+1.5); the recognised text
  starts at 1 (+1 if the publisher heard it too)
- cleanup edits need only to fit the audio (deletions of fillers always pass)
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

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
ACCEPT_PER_WORD = 10.0
CONTEXT_SECONDS = 3.0
PAD_SECONDS = 0.15
MAX_SPAN_TOKENS = 6
# A glossary term counts as known once it has this much evidence: a confirmed
# speaker or publisher-agreed correction adds 2, an LLM-only correction adds 1
TRUSTED_COUNT = 2
CONFIDENCE_SUPPORT = {"high": 2.0, "medium": 1.0, "low": 0.5}
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


def _name_part(text: str) -> str:
    """The capitalised run of a phrase: "Stephen Seiler has" -> "Stephen Seiler"."""
    run, best = [], []
    for w in (_EDGE_PUNCT.sub("", t) for t in text.split()):
        if w[:1].isupper():
            run.append(w)
            best = max(best, run, key=len)
        else:
            run = []
    return " ".join(best)


def _looks_like_name(text: str) -> bool:
    return any(w[:1].isupper() for w in text.split()[1:]) or (text[:1].isupper() and len(text.split()) == 1)


def load_audio(path: Path) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(path)).set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
    return np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0


# --- Glossary ---


def glossary_path(storage: Path, show: str) -> Path:
    safe = re.sub(r"[^\w\s-]", "", show).strip() or "default"
    return storage / "glossaries" / f"{safe}.json"


def load_glossary(path: Path) -> Dict[str, int]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_glossary(path: Path, glossary: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = dict(sorted(glossary.items(), key=lambda kv: (-kv[1], kv[0])))
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
        glossary: Optional[Dict[str, int]] = None,
        accept_per_word: float = ACCEPT_PER_WORD,
    ):
        self.lines = build_lines(segments)
        self.audio = audio
        self.scorer = scorer
        self.reference = sorted(reference or [], key=lambda w: w.start)
        self._ref_starts = [w.start for w in self.reference]
        self.glossary = dict(glossary or {})
        self.accept = accept_per_word
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
        """Glossary terms with independent evidence (count >= TRUSTED_COUNT)."""
        return [t for t, n in self.glossary.items() if n >= TRUSTED_COUNT]

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
                        out.append(Suspect(li, phrase, [lowered[match[0]]], "low"))
        return out

    def _find(self, line: Line, text: str) -> Optional[Tuple[int, int]]:
        target = [_key(t) for t in text.split() if _key(t)]
        keys = [_key(t) for t, _ in line.tokens()]
        if not target:
            return None
        for i in range(len(keys) - len(target) + 1):
            if keys[i:i + len(target)] == target:
                return i, i + len(target)
        return None

    def resolve(self, suspects: List[Suspect], vocabulary: str = "") -> List[Correction]:
        """Score every suspect's candidates against the audio and apply the winners."""
        # Spellings of names we trust: capitalised words of the vocabulary + glossary terms
        known = {_key(w) for w in re.findall(r"[\w'’-]+", vocabulary) if w[:1].isupper()}
        known |= {k for term in self.trusted_terms() for k in _words(term)}
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
                    match = difflib.get_close_matches(_key(core), list(known_words), n=1, cutoff=0.75)
                    if match:
                        w = w.replace(core, known_words[match[0]])
                out.append(w)
            return " ".join(out)

        # group by (line, span); merge candidates from several proposers
        spans: Dict[Tuple[int, int, int], Dict[str, Tuple[str, float]]] = {}
        for s in suspects + self.glossary_suspects():
            if not (0 <= s.line < len(self.lines)):
                continue
            found = self._find(self.lines[s.line], s.text)
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
            if theirs and theirs_key != [_key(t) for t in original_tokens] and len(theirs_key) <= MAX_SPAN_TOKENS:
                src, sup = cands.get(theirs, ("publisher", 0.0))
                cands[theirs] = (src, sup)
            if not cands:
                continue

            names = list(cands)
            cleaned = [_clean_candidate(c.split(), original_tokens) for c in names]
            penalties = self.penalties(line, t1, t2, cleaned)

            def support(text: str, base: float) -> float:
                s = base
                if theirs_key and [_key(w) for w in text.split() if _key(w)] == theirs_key:
                    s += 1.0
                capitalised = [_key(w) for w in text.split() if w[:1].isupper() and _key(w)]
                if capitalised and all(k in known for k in capitalised):
                    s += 1.5
                return s

            original_support = support(original, 1.0)  # what was recognised starts at 1
            # (support, tie-break on fit to the audio, ...)
            options = [
                (support(name, cands[name][1]), -pen, name, cands[name][0], pen, clean)
                for name, pen, clean in zip(names, penalties, cleaned)
                if pen <= self.accept
            ]
            if not options:
                continue
            score, _, name, source, pen, clean = max(options)
            alternatives = [o[2] for o in sorted(options, reverse=True) if o[2] != name]
            if score > original_support:
                confidence = "high" if score - original_support >= 1.0 else "medium"
                line.replace(t1, t2, clean)
                agreed = bool(theirs_key) and [_key(w) for w in clean if _key(w)] == theirs_key
                corrections.append(
                    Correction(start, original, " ".join(clean), source, pen, confidence, alternatives, agreed)
                )
            elif score >= original_support - 0.25 and (_looks_like_name(original) or _looks_like_name(name)):
                corrections.append(Correction(start, original, original, "kept", 0.0, "low", [name] + alternatives))
        corrections.sort(key=lambda c: c.time)
        self.corrections = corrections
        applied = [c for c in corrections if c.confidence != "low"]
        logger.info(f"Verification: {len(applied)} corrections applied, "
                    f"{len(corrections) - len(applied)} left for review, {len(spans)} spans checked")
        return corrections

    # --- Cleanup edits ---

    def check_cleanup(self, line_offset: int, original_chunk: str, cleaned_chunk: str) -> Tuple[str, List[dict]]:
        """
        Apply the cleanup's edits to lines [line_offset, ...) where the audio supports them.
        Returns the resulting chunk text and the rejected edits.
        """
        rejected = []
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
                if op == "delete" or same_letters:
                    accepted.append((i1, i2, new[j1:j2]))
                else:
                    to_score.append((i1, i2, new[j1:j2]))
            for i1, i2, repl in to_score:
                pen = self.penalties(line, i1, i2, [repl])[0]
                if pen <= self.accept:
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

    def learn(self, speaker_names: Sequence[str] = (), verified: Sequence[str] = ()) -> Dict[str, int]:
        """
        Add evidence for names to the glossary (returned; caller saves it). Publisher-confirmed
        speakers and corrections the publisher agrees with count 2 (trusted at once); other
        speaker names and high-confidence corrections count 1, so they become trusted only
        when another episode confirms them.
        """
        lowercase_words = {w for line in self.lines for w in re.findall(r"\b[a-z][\w']*", line.content())}

        def add(term: str, weight: int) -> None:
            term = re.sub(r"['’]s$", "", term)
            # a lone capitalised word that also appears in lowercase is just a sentence start
            if not term or (" " not in term and term.lower() in lowercase_words):
                return
            self.glossary[term] = self.glossary.get(term, 0) + weight

        for c in self.corrections:
            if c.confidence == "high":
                add(_name_part(c.chosen), 2 if c.publisher else 1)
        for name in set(speaker_names):
            if name and " " in name.strip():
                add(name.strip(), 2 if name in verified else 1)
        return self.glossary
