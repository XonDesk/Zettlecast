"""
Transcript Enhancement Module

LLM (Ollama) post-processing of a diarized transcript:
1. analyze(): one structured call over the whole transcript for keywords,
   summary, key points, chapters and the real names behind speaker labels
2. cleanup_transcript(): line-preserving fix-up of ASR errors and fillers,
   with uncertain corrections marked [[text??]]
"""

import difflib
import logging
import re
from typing import Callable, Dict, List, Optional, Tuple

from ..llm import LLMError, chat, chat_each

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are analysing a diarized podcast transcript.
Each line is "[start - end] speaker_N: text" with times in seconds.

Podcast: {podcast}
Episode: {episode}
{show_notes}{known_speakers}
Return JSON with:
- "keywords": 5-10 lowercase topic keywords: the subjects, concepts and methods discussed, in their
  most common general form (e.g. "heat training", "ultrarunning"). No names of people or organisations.
- "people": full names of the people who speak or are discussed in some depth (not passing mentions),
  spelled as in the show notes when they name them
- "organisations": named organisations, institutions, companies, products, races and events
  discussed in some depth (e.g. "UTMB", "TrainingPeaks", "NTNU")
- "summary": 3-5 sentences covering the topic, who the guests are and the key conclusions.
  Third person. Do not start with "In this episode" or "This podcast".
- "key_points": 5-10 specific, concrete takeaways (e.g. "Training the gut with 60-90 g carbs/hour
  reduces GI distress on race day", not "Gut training was discussed")
- "chapters": the episode's sections in order, each {{"name", "start_time" (seconds, from the
  transcript), "description"}}; include intro, sponsor reads and outro when present
- "speakers": for each speaker label, the person's name if the transcript makes it clear
  (introductions, being addressed by name), otherwise null. When the show notes name the
  person, use the show notes' spelling (speech recognition often misspells names). When someone
  is only called by a first name or title (e.g. "Coach Adam") and matches a person from
  earlier episodes, use that full name. Example: {{"speaker_0": "Jason Koop", "speaker_1": null}}

Transcript:
{transcript}"""

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}},
        "people": {"type": "array", "items": {"type": "string"}},
        "organisations": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "start_time": {"type": "number"},
                    "description": {"type": "string"},
                },
                "required": ["name", "start_time", "description"],
            },
        },
        "speakers": {
            "type": "object",
            "additionalProperties": {"type": ["string", "null"]},
        },
    },
    "required": ["keywords", "people", "organisations", "summary", "key_points", "chapters", "speakers"],
}

CLEANUP_PROMPT = """You are cleaning up an automated podcast transcript.

Fix these issues while preserving timestamps and speaker labels:
1. Fix domain-specific terms misrecognized by speech-to-text:
   - Scientific terms: "exercise physiology" not "exercise astrology"
   - Technical terms: "VO2 max" not "vo two max", "lactate threshold" not "lactaid threshold"
   - Names of people, places, universities, and organizations{names_hint}
2. Remove filler words: um, uh, like, you know, I mean (unless meaningful)
3. Fix obvious transcription errors using surrounding context
4. Keep every line's "[start - end] speaker:" prefix exactly as it is, one output line per input line
5. Do NOT change meaning, add content, or rephrase sentences

Never swap a person's name for a different, better-known person. Only correct a name
to one in the correct spellings below or one you are certain of; otherwise leave it.

IMPORTANT: If you change a word or phrase but are less than 80% confident
the correction is right, wrap it in double brackets: [[corrected text??]]
Only use this for uncertain corrections, not obvious fixes.

Return ONLY the cleaned transcript lines. No commentary.

{vocabulary}Transcript chunk:
{chunk}"""

SUSPECTS_PROMPT = """You are checking an automatic podcast transcript for misrecognised words.
Speech recognition often mishears names and uncommon terms as similar-sounding words.

{context}{hints}
List the words or phrases in the numbered lines below that were probably misheard, especially
names of people, places, organisations, products, races and technical terms, and phrases that
make no sense in context. For each give:
- "line": the line number
- "text": the words exactly as they appear in that line (1-6 words)
- "candidates": 1-3 corrections that SOUND LIKE the original. Use real names you know when they
  fit the context (e.g. a known athlete, scientist or company mentioned with matching details).
- "confidence": "high" if you know the correct term, "medium" if likely, "low" if a guess
- "about": for a name, 2-5 words saying who or what it is, taken from the conversation, to search
  an encyclopedia with (e.g. "Norwegian triathlete", "ultramarathon in California",
  "exercise physiologist"); "" for other words

Only list likely errors; most words are right. Return {{"suspects": []}} if there are none.

Lines:
{chunk}"""

SUSPECTS_SCHEMA = {
    "type": "object",
    "properties": {
        "suspects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "integer"},
                    "text": {"type": "string"},
                    "candidates": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "about": {"type": "string"},
                },
                "required": ["line", "text", "candidates", "confidence", "about"],
            },
        }
    },
    "required": ["suspects"],
}

# Regex for parsing uncertainty markers from LLM output
_UNCERTAIN_MARKER_RE = re.compile(r"\[\[(.+?)\?\?\]\]")
_LINE_PREFIX_RE = re.compile(r"^\[[\d.]+s - [\d.]+s\] [^:]+:")

# ~25k tokens; an hour-long episode is ~60k characters
MAX_ANALYSIS_CHARS = 100_000
MAX_SHOW_NOTES_CHARS = 4_000
# Repeated in every cleanup chunk, so kept short
MAX_VOCABULARY_CHARS = 2_000


_SENTENCE_END = (".", "?", "!", ":", "]")
_PRONOUN_I = {"i", "i'm", "i've", "i'll", "i'd"}


def _key(token: str) -> str:
    """Comparison form of a token: lowercase word characters only."""
    return re.sub(r"[^\w']", "", token.replace("[[", "").replace("??]]", "").lower().replace("’", "'"))


def _words(text: str) -> set:
    return {_key(w) for w in re.findall(r"[\w'’]+", text)}


def _introduces_name(tokens: List[str], j: int, original: set, allowed: set) -> bool:
    """A capitalised word mid-sentence that is neither in the original line nor allowed."""
    word = re.sub(r"^[^\w]+", "", tokens[j].replace("[[", ""))
    if not word or not word[0].isupper():
        return False
    if j == 0 or tokens[j - 1].endswith(_SENTENCE_END):
        return False
    key = _key(word)
    return bool(key) and key not in original and key not in allowed and key not in _PRONOUN_I


def guard_names(original: str, cleaned: str, allowed: set) -> Tuple[str, List[Dict]]:
    """
    Undo cleanup edits that bring in a name (a capitalised word) found neither in
    the original line nor in the allowed vocabulary, so the LLM can't swap a misheard
    name for a better-known person. Reverted edits are returned for review.
    Joining split words ("Koop Cast" -> "KoopCast") is always kept.
    """
    lines, reverted = [], []
    for orig_line, new_line in zip(original.split("\n"), cleaned.split("\n")):
        ot, ct = orig_line.split(" "), new_line.split(" ")
        orig_words = {_key(t) for t in ot}
        matcher = difflib.SequenceMatcher(a=[_key(t) for t in ot], b=[_key(t) for t in ct], autojunk=False)
        out: List[str] = []
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            new = ct[j1:j2]
            same_letters = "".join(map(_key, ot[i1:i2])) == "".join(map(_key, new))
            if op in ("replace", "insert") and not same_letters and any(
                _introduces_name(ct, j, orig_words, allowed) for j in range(j1, j2)
            ):
                out += ot[i1:i2]
                reverted.append({"text": f"{' '.join(ot[i1:i2])} (cleanup suggested: {' '.join(new)})"})
            else:
                out += new
        lines.append(" ".join(out))
    return "\n".join(lines), reverted


def snap_name(name: str, known: List[str], cutoff: float = 0.85) -> str:
    """'Ryan Anderson' -> 'Ryne Anderson' when a near-identical known name exists."""
    match = difflib.get_close_matches(name, known, n=1, cutoff=cutoff)
    return match[0] if match else name


class TranscriptEnhancer:
    """Enhance transcripts using the configured Ollama model."""

    @staticmethod
    def extract_uncertain_corrections(text: str) -> Tuple[str, List[Dict]]:
        """
        Parse [[text??]] uncertainty markers from LLM cleanup output.

        Returns:
            Tuple of:
              - text with markers removed (correction text kept)
              - list of uncertain items, each with "text" and "position" keys
        """
        uncertain_items = []
        offset_adjustment = 0

        for match in _UNCERTAIN_MARKER_RE.finditer(text):
            uncertain_items.append({
                "text": match.group(1),
                "position": match.start() - offset_adjustment,
            })
            # [[text??]] -> text removes 6 characters
            offset_adjustment += 6

        cleaned = _UNCERTAIN_MARKER_RE.sub(r"\1", text)
        if uncertain_items:
            logger.info(f"Found {len(uncertain_items)} uncertain corrections")
        return cleaned, uncertain_items

    @staticmethod
    def _is_valid_cleanup_response(original: str, response: str) -> bool:
        """
        A cleaned chunk must keep one line per input line with identical
        "[start - end] speaker:" prefixes and roughly the same length.
        """
        if len(response) < len(original) * 0.5:
            return False
        orig_lines = [l for l in original.split("\n") if l.strip()]
        new_lines = [l for l in response.split("\n") if l.strip()]
        if len(orig_lines) != len(new_lines):
            return False
        for a, b in zip(orig_lines, new_lines):
            prefix = _LINE_PREFIX_RE.match(a)
            if prefix and not b.startswith(prefix.group(0)):
                return False
        return True

    @staticmethod
    def _split_lines(transcript: str, chunk_size: int) -> List[str]:
        chunks, current, length = [], [], 0
        for line in transcript.split("\n"):
            if current and length + len(line) > chunk_size:
                chunks.append("\n".join(current))
                current, length = [], 0
            current.append(line)
            length += len(line) + 1
        if current:
            chunks.append("\n".join(current))
        return chunks

    def cleanup_transcript(
        self,
        transcript: str,
        chunk_size: int = 8000,
        speaker_names: Optional[Dict[str, str]] = None,
        vocabulary: str = "",
        checker: Optional[Callable[[int, str, str], Tuple[str, List[Dict]]]] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Clean up the transcript chunk by chunk.

        Chunks the model mangles (lost lines or prefixes) are kept as-is.

        Args:
            vocabulary: Correct spellings to follow (podcast name, people, show notes)
            checker: (line offset, original chunk, cleaned chunk) -> (accepted chunk, rejected
                edits); defaults to guard_names. The verifier passes an audio-based check.

        Returns:
            (cleaned transcript, uncertain corrections)
        """
        if not transcript.strip():
            return transcript, []

        names = [n for n in (speaker_names or {}).values() if n]
        names_hint = f"\n   - Speakers in this episode: {', '.join(names)}" if names else ""
        vocab = vocabulary.strip()[:MAX_VOCABULARY_CHARS]
        vocab_block = (
            "Correct spellings for this episode (use them where the transcript has a misheard "
            f"version, e.g. of the podcast's or a person's name):\n{vocab}\n\n"
            if vocab else ""
        )
        allowed = _words(vocabulary) | _words(" ".join(names))

        chunks = self._split_lines(transcript, chunk_size)
        logger.info(f"Cleaning transcript in {len(chunks)} chunks")

        cleaned_chunks: List[str] = []
        all_uncertain: List[Dict] = []
        running_offset = 0
        line_offset = 0
        review_reverted = checker is None  # without audio, reverted names go to review
        if checker is None:
            def checker(_offset: int, original: str, cleaned: str) -> Tuple[str, List[Dict]]:
                return guard_names(original, cleaned, allowed)

        # the LLM cleans several chunks at once; checking stays in order, one chunk at a time
        responses = chat_each(
            lambda chunk=chunk: chat(CLEANUP_PROMPT.format(chunk=chunk, names_hint=names_hint, vocabulary=vocab_block))
            for chunk in chunks
        )
        for i, (chunk, result) in enumerate(zip(chunks, responses)):
            try:
                if isinstance(result, Exception):
                    raise result
                result = result.strip()
                if self._is_valid_cleanup_response(chunk, result):
                    # take the [[x??]] markers off before checking, keep the ones that survive
                    unmarked, marked = self.extract_uncertain_corrections(result)
                    result, reverted = checker(line_offset, chunk, unmarked)
                    if reverted:
                        logger.info(f"Kept the original for {len(reverted)} cleanup edit(s)")
                    for item in marked:
                        pos = result.find(item["text"])
                        if pos >= 0:
                            all_uncertain.append({"text": item["text"], "position": running_offset + pos})
                    if review_reverted:
                        all_uncertain.extend({**r, "position": running_offset} for r in reverted)
                    cleaned_chunks.append(result)
                else:
                    logger.warning(f"Cleanup chunk {i + 1} changed the line structure; keeping original")
                    cleaned_chunks.append(chunk)
            except LLMError as e:
                logger.warning(f"Cleanup failed for chunk {i + 1}: {e}")
                cleaned_chunks.append(chunk)
            running_offset += len(cleaned_chunks[-1]) + 1
            line_offset += chunk.count("\n") + 1

        return "\n".join(cleaned_chunks), all_uncertain

    def find_suspects(
        self,
        transcript: str,
        context: str = "",
        hints: Optional[List[Tuple[int, str, str]]] = None,
        chunk_size: int = 8000,
    ) -> List[Dict]:
        """
        LLM pass listing probably-misheard words with sound-alike candidates.

        Args:
            transcript: Lines to check; suspects refer to them by 0-based line number
            context: Podcast, people, show notes and known terms
            hints: (line, ours, theirs) where another transcript heard something else
        """
        by_line: Dict[int, List[Tuple[str, str]]] = {}
        for li, ours, theirs in hints or []:
            by_line.setdefault(li, []).append((ours, theirs))

        spans, prompts = [], []
        start = 0
        for chunk in self._split_lines(transcript, chunk_size):
            chunk_lines = chunk.split("\n")
            end = start + len(chunk_lines)
            numbered = "\n".join(f"{start + k}| {line}" for k, line in enumerate(chunk_lines))
            chunk_hints = [
                f"{li}| ours: {o} | theirs: {t}" for li in range(start, end) for o, t in by_line.get(li, [])
            ][:60]
            hint_block = (
                "A second, independent transcript heard these passages differently "
                "(line| ours | theirs); either may be wrong:\n" + "\n".join(chunk_hints) + "\n"
                if chunk_hints else ""
            )
            spans.append((start, end))
            prompts.append(SUSPECTS_PROMPT.format(context=context, hints=hint_block, chunk=numbered))
            start = end

        suspects: List[Dict] = []
        results = chat_each(lambda p=p: chat(p, schema=SUSPECTS_SCHEMA) for p in prompts)
        for (start, end), result in zip(spans, results):
            try:
                if isinstance(result, Exception):
                    raise result
                suspects += [s for s in result.get("suspects", []) if start <= int(s.get("line", -1)) < end]
            except (LLMError, ValueError, TypeError) as e:
                logger.warning(f"Suspect search failed for lines {start}-{end}: {e}")
        return suspects

    def analyze(
        self,
        transcript: str,
        podcast: str = "",
        episode: str = "",
        show_notes: str = "",
        known_speakers: Optional[List[str]] = None,
        published_speakers: Optional[Dict[str, str]] = None,
    ) -> dict:
        """Keywords, summary, key points, chapters and speaker names in one call."""
        if len(transcript) > MAX_ANALYSIS_CHARS:
            half = MAX_ANALYSIS_CHARS // 2
            transcript = transcript[:half] + "\n[...]\n" + transcript[-half:]
        notes = show_notes.strip()[:MAX_SHOW_NOTES_CHARS]
        try:
            result = chat(
                ANALYSIS_PROMPT.format(
                    podcast=podcast or "unknown",
                    episode=episode or "unknown",
                    show_notes=f"\nShow notes (from the podcast feed):\n{notes}\n" if notes else "",
                    known_speakers=(
                        f"\nPeople on earlier episodes of this show: {', '.join(known_speakers)}\n"
                        if known_speakers else ""
                    ) + (
                        "\nSpeakers as named by the publisher (use exactly these names): "
                        + ", ".join(f"{k} = {v}" for k, v in published_speakers.items()) + "\n"
                        if published_speakers else ""
                    ),
                    transcript=transcript,
                ),
                schema=ANALYSIS_SCHEMA,
            )
        except LLMError as e:
            logger.error(f"Transcript analysis failed: {e}")
            return {"keywords": [], "people": [], "organisations": [], "summary": "", "key_points": [],
                    "chapters": [], "speakers": {}}

        result["keywords"] = [k.strip().lower() for k in result.get("keywords", []) if k.strip()]
        for field in ("people", "organisations"):
            result[field] = list(dict.fromkeys(v.strip() for v in result.get(field) or [] if v and v.strip()))
        result["speakers"] = {k: v.strip() for k, v in (result.get("speakers") or {}).items() if v and v.strip()}
        return result

    @staticmethod
    def apply_speaker_names(transcript: str, speakers: Dict[str, str]) -> str:
        """Replace "speaker_N:" labels with names (only where a name was identified)."""
        for label, name in speakers.items():
            transcript = re.sub(rf"(\] ){re.escape(label)}:", rf"\g<1>{name}:", transcript)
        return transcript

    def enhance(
        self,
        transcript: str,
        podcast: str = "",
        episode: str = "",
        cleanup: bool = True,
        show_notes: str = "",
        known_speakers: Optional[List[str]] = None,
        published_speakers: Optional[Dict[str, str]] = None,
        verifier=None,
    ) -> dict:
        """
        Run the full enhancement pipeline.

        published_speakers (label -> name, from the publisher's transcript) take
        precedence over the names the LLM infers.

        With a verifier (podcast.verifier.TranscriptVerifier, which holds the timed
        words and the audio), likely misheard words are resolved against the audio
        and every cleanup edit must fit the audio too.

        Returns:
            Dict with cleaned_transcript, keywords, sections, summary, key_points,
            speakers, verified_speakers, uncertain_corrections, corrections, verification
        """
        analysis = self.analyze(
            transcript,
            podcast=podcast,
            episode=episode,
            show_notes=show_notes,
            known_speakers=known_speakers,
            published_speakers=published_speakers,
        )
        published_speakers = published_speakers or {}
        # Snap near-misses to known spellings (publisher's names first), and carry
        # the correction into the summary, key points, chapters and keywords
        known = list(dict.fromkeys([*published_speakers.values(), *(known_speakers or [])]))
        speakers, renames = {}, {}
        for label, name in analysis.get("speakers", {}).items():
            fixed = published_speakers.get(label) or snap_name(name, known)
            speakers[label] = fixed
            if fixed != name:
                renames[name] = fixed
        speakers.update(published_speakers)
        if renames:
            logger.info(f"Speaker names corrected: {renames}")
            analysis = self._rename(analysis, renames)

        people = list(dict.fromkeys([*speakers.values(), *(known_speakers or [])]))
        # only terms with independent evidence: the vocabulary also counts as known spellings
        glossary_terms = verifier.trusted_terms()[:80] if verifier is not None else []
        vocabulary = "\n".join(filter(None, [
            f"Podcast: {podcast}" if podcast else "",
            f"Episode: {episode}" if episode else "",
            f"People: {', '.join(people)}" if people else "",
            f"Names and terms from earlier episodes: {', '.join(glossary_terms)}" if glossary_terms else "",
            f"Show notes: {show_notes.strip()}" if show_notes.strip() else "",
        ]))

        corrections = []
        if verifier is not None:
            from .verifier import Suspect

            found = self.find_suspects(
                verifier.text(), context=vocabulary[:MAX_VOCABULARY_CHARS] + "\n", hints=verifier.publisher_hints()
            )
            corrections = verifier.resolve(
                [Suspect(int(s["line"]), s["text"], list(s.get("candidates") or []),
                         s.get("confidence", "medium"), s.get("about") or "")
                 for s in found],
                vocabulary=vocabulary,
            )
            # without edge punctuation, so "Ryan Anderson," also fixes "Ryan Anderson's Role"
            name_fixes = {
                c.original.strip(" ,.;:!?\"'"): c.chosen.strip(" ,.;:!?\"'") for c in corrections
                if c.confidence != "low" and c.original != c.chosen and any(w[:1].isupper() for w in c.chosen.split())
            }
            if name_fixes:
                analysis = self._rename(analysis, name_fixes)
            transcript = verifier.text()

        cleaned, uncertain = transcript, []
        if cleanup:
            cleaned, uncertain = self.cleanup_transcript(
                transcript, speaker_names=speakers, vocabulary=vocabulary,
                checker=verifier.check_cleanup if verifier is not None else None,
            )

        # graph links need one spelling per person: snap to speakers and confirmed glossary names
        spellings = [*people, *glossary_terms]
        return {
            "cleaned_transcript": self.apply_speaker_names(cleaned, speakers),
            "keywords": analysis.get("keywords", []),
            # a first name alone ("Fred") can't tell people apart
            "people": list(dict.fromkeys(snap_name(p, spellings) for p in analysis.get("people", []) if " " in p)),
            "organisations": list(dict.fromkeys(snap_name(o, glossary_terms) for o in analysis.get("organisations", []))),
            "sections": analysis.get("chapters", []),
            "summary": analysis.get("summary", ""),
            "key_points": analysis.get("key_points", []),
            "speakers": speakers,
            "verified_speakers": sorted(set(published_speakers.values())),
            "uncertain_corrections": uncertain,
            "corrections": [c.as_dict() for c in corrections],
            "verification": {
                "corrections": sum(c.confidence != "low" for c in corrections),
                "review": sum(c.confidence == "low" for c in corrections),
                "cleanup_edits_kept": verifier.accepted_edits,
                "cleanup_edits_rejected": len(verifier.rejected),
                "names_looked_up": verifier.lookups,
            } if verifier is not None else {},
        }

    @staticmethod
    def _rename(analysis: dict, renames: Dict[str, str]) -> dict:
        """Apply name corrections to the analysis text fields."""
        def fix(text: str) -> str:
            for old, new in renames.items():
                text = text.replace(old, new).replace(old.lower(), new.lower())
            return text

        return {
            **analysis,
            "summary": fix(analysis.get("summary", "")),
            "key_points": [fix(p) for p in analysis.get("key_points", [])],
            "keywords": [fix(k) for k in analysis.get("keywords", [])],
            "people": [fix(p) for p in analysis.get("people", [])],
            "organisations": [fix(o) for o in analysis.get("organisations", [])],
            "chapters": [
                {**c, "name": fix(c.get("name", "")), "description": fix(c.get("description", ""))}
                for c in analysis.get("chapters", [])
            ],
        }
