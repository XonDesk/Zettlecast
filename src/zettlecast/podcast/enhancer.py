"""
Transcript Enhancement Module

LLM (Ollama) post-processing of a diarized transcript:
1. analyze(): one structured call over the whole transcript for keywords,
   summary, key points, chapters and the real names behind speaker labels
2. cleanup_transcript(): line-preserving fix-up of ASR errors and fillers,
   with uncertain corrections marked [[text??]]
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from ..llm import LLMError, chat

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are analysing a diarized podcast transcript.
Each line is "[start - end] speaker_N: text" with times in seconds.

Podcast: {podcast}
Episode: {episode}

Return JSON with:
- "keywords": 5-10 lowercase topic keywords (main topics, concepts, notable people or organisations)
- "summary": 3-5 sentences covering the topic, who the guests are and the key conclusions.
  Third person. Do not start with "In this episode" or "This podcast".
- "key_points": 5-10 specific, concrete takeaways (e.g. "Training the gut with 60-90 g carbs/hour
  reduces GI distress on race day", not "Gut training was discussed")
- "chapters": the episode's sections in order, each {{"name", "start_time" (seconds, from the
  transcript), "description"}}; include intro, sponsor reads and outro when present
- "speakers": for each speaker label, the person's name if the transcript makes it clear
  (introductions, being addressed by name), otherwise null. Example: {{"speaker_0": "Jason Koop", "speaker_1": null}}

Transcript:
{transcript}"""

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}},
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
    "required": ["keywords", "summary", "key_points", "chapters", "speakers"],
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

IMPORTANT: If you change a word or phrase but are less than 80% confident
the correction is right, wrap it in double brackets: [[corrected text??]]
Only use this for uncertain corrections, not obvious fixes.

Return ONLY the cleaned transcript lines. No commentary.

Transcript chunk:
{chunk}"""

# Regex for parsing uncertainty markers from LLM output
_UNCERTAIN_MARKER_RE = re.compile(r"\[\[(.+?)\?\?\]\]")
_LINE_PREFIX_RE = re.compile(r"^\[[\d.]+s - [\d.]+s\] [^:]+:")

# ~25k tokens; an hour-long episode is ~60k characters
MAX_ANALYSIS_CHARS = 100_000


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
    ) -> Tuple[str, List[Dict]]:
        """
        Clean up the transcript chunk by chunk.

        Chunks the model mangles (lost lines or prefixes) are kept as-is.

        Returns:
            (cleaned transcript, uncertain corrections)
        """
        if not transcript.strip():
            return transcript, []

        names = [n for n in (speaker_names or {}).values() if n]
        names_hint = f"\n   - Speakers in this episode: {', '.join(names)}" if names else ""

        chunks = self._split_lines(transcript, chunk_size)
        logger.info(f"Cleaning transcript in {len(chunks)} chunks")

        cleaned_chunks: List[str] = []
        all_uncertain: List[Dict] = []
        running_offset = 0

        for i, chunk in enumerate(chunks):
            try:
                result = chat(CLEANUP_PROMPT.format(chunk=chunk, names_hint=names_hint)).strip()
                if self._is_valid_cleanup_response(chunk, result):
                    result, uncertain = self.extract_uncertain_corrections(result)
                    for item in uncertain:
                        item["position"] += running_offset
                    all_uncertain.extend(uncertain)
                    cleaned_chunks.append(result)
                else:
                    logger.warning(f"Cleanup chunk {i + 1} changed the line structure; keeping original")
                    cleaned_chunks.append(chunk)
            except LLMError as e:
                logger.warning(f"Cleanup failed for chunk {i + 1}: {e}")
                cleaned_chunks.append(chunk)
            running_offset += len(cleaned_chunks[-1]) + 1

        return "\n".join(cleaned_chunks), all_uncertain

    def analyze(self, transcript: str, podcast: str = "", episode: str = "") -> dict:
        """Keywords, summary, key points, chapters and speaker names in one call."""
        if len(transcript) > MAX_ANALYSIS_CHARS:
            half = MAX_ANALYSIS_CHARS // 2
            transcript = transcript[:half] + "\n[...]\n" + transcript[-half:]
        try:
            result = chat(
                ANALYSIS_PROMPT.format(
                    podcast=podcast or "unknown",
                    episode=episode or "unknown",
                    transcript=transcript,
                ),
                schema=ANALYSIS_SCHEMA,
            )
        except LLMError as e:
            logger.error(f"Transcript analysis failed: {e}")
            return {"keywords": [], "summary": "", "key_points": [], "chapters": [], "speakers": {}}

        result["keywords"] = [k.strip().lower() for k in result.get("keywords", []) if k.strip()]
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
    ) -> dict:
        """
        Run the full enhancement pipeline.

        Returns:
            Dict with cleaned_transcript, keywords, sections, summary,
            key_points, speakers, uncertain_corrections
        """
        analysis = self.analyze(transcript, podcast=podcast, episode=episode)
        speakers = analysis.get("speakers", {})

        cleaned, uncertain = transcript, []
        if cleanup:
            cleaned, uncertain = self.cleanup_transcript(transcript, speaker_names=speakers)

        return {
            "cleaned_transcript": self.apply_speaker_names(cleaned, speakers),
            "keywords": analysis.get("keywords", []),
            "sections": analysis.get("chapters", []),
            "summary": analysis.get("summary", ""),
            "key_points": analysis.get("key_points", []),
            "speakers": speakers,
            "uncertain_corrections": uncertain,
        }
