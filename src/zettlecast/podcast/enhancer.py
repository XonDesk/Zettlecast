"""
Transcript Enhancement Module

LLM-based cleanup, keyword extraction, section detection, summarization,
and key point extraction using Ollama.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


CLEANUP_PROMPT = """You are cleaning up an automated podcast transcript.

Fix these issues while preserving timestamps and speaker labels:
1. Fix domain-specific terms misrecognized by speech-to-text:
   - Scientific terms: "exercise physiology" not "exercise astrology"
   - Technical terms: "VO2 max" not "vo two max", "lactate threshold" not "lactaid threshold"
   - Names of people, places, universities, and organizations
2. Remove filler words: um, uh, like, you know, I mean (unless meaningful)
3. Fix obvious transcription errors using surrounding context
4. Keep speaker labels intact (speaker_0, Speaker 1, SPEAKER_00, etc.)
5. Do NOT change meaning, add content, or rephrase sentences

IMPORTANT: If you change a word or phrase but are less than 80% confident
the correction is right, wrap it in double brackets: [[corrected text??]]
Only use this for uncertain corrections, not obvious fixes.

Return ONLY the cleaned transcript text. No commentary.

Transcript chunk:
{chunk}"""

KEYWORD_PROMPT = """Extract 5-10 keywords from this podcast transcript.
Focus on: main topics discussed, key concepts, notable people or organizations mentioned.
Return ONLY a JSON array of strings, nothing else.

Example: ["gut training", "carbohydrate absorption", "ultra marathon nutrition"]

Transcript (first 6000 characters):
{transcript}"""

SECTION_PROMPT = """Identify chapters/sections in this podcast transcript.
Common sections: introduction, main topic discussion, sponsor reads, Q&A, conclusion/outro.

Return ONLY a JSON array of objects with "name", "start_time", and "description" fields.
Times should be in seconds (numbers, not strings).

Example: [{{"name": "Introduction", "start_time": 0.0, "description": "Hosts introduce the topic"}}]

Transcript:
{transcript}"""

SUMMARY_PROMPT = """Write a concise summary (3-5 sentences) of this podcast transcript.
Cover: the main topic discussed, who the guests are, and the key conclusions or insights.
Write in third person. Do NOT start with "In this episode" or "This podcast".

Return ONLY the summary text, no headers or labels.

Transcript:
{transcript}"""

KEY_POINTS_PROMPT = """Extract 5-10 key takeaways from this podcast transcript.
Each point should be a specific, concrete insight - not a vague topic reference.

Good example: "Training the gut with 60-90g carbs/hour during long runs can reduce GI distress on race day"
Bad example: "Gut training was discussed"

Return ONLY a JSON array of strings, nothing else.

Example: ["Gut training requires structured, repetitive carbohydrate intake during exercise", "Only 8 studies on gut training existed at the time of the meta-analysis"]

Transcript:
{transcript}"""

# Regex for parsing uncertainty markers from LLM output
_UNCERTAIN_MARKER_RE = re.compile(r"\[\[(.+?)\?\?\]\]")


class TranscriptEnhancer:
    """
    Enhance transcripts using LLM (Ollama) for cleanup and metadata extraction.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = None,
    ):
        self.base_url = ollama_base_url
        self.model = model or settings.ollama_model or "llama3.2:3b"

    @staticmethod
    def unload_model(base_url: str, model: str):
        """
        Unload the model from Ollama memory to free VRAM.
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{base_url}/api/generate",
                    json={"model": model, "keep_alive": 0}
                )
                if resp.status_code == 200:
                    logger.info(f"Unloaded Ollama model: {model}")
                else:
                    logger.warning(f"Failed to unload model {model}: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Error unloading Ollama model: {e}")

    async def _generate(self, prompt: str) -> str:
        """Call Ollama generate endpoint."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["response"]

    def _generate_sync(self, prompt: str) -> str:
        """Synchronous version of generate."""
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["response"]

    def _is_valid_cleanup_response(self, original: str, response: str) -> bool:
        """
        Check if LLM response is a valid cleaned transcript.

        Detects cases where LLM returns placeholder text or conversational
        responses instead of the actual cleaned transcript.
        """
        # Response should be at least 50% of original length
        if len(response) < len(original) * 0.5:
            return False

        # Detect placeholder patterns that indicate LLM didn't understand the task
        placeholder_patterns = [
            "I'll need to see",
            "Please provide",
            "I don't see",
            "I can help you",
            "provide the entire",
            "I'd be happy to",
            "Here is the cleaned",
            "Here's the cleaned",
            "I've cleaned",
        ]
        return not any(p.lower() in response.lower() for p in placeholder_patterns)

    @staticmethod
    def extract_uncertain_corrections(text: str) -> Tuple[str, List[Dict]]:
        """
        Parse [[text??]] uncertainty markers from LLM cleanup output.

        The cleanup prompt instructs the LLM to wrap uncertain corrections
        in [[corrected text??]] markers. This method extracts them.

        Args:
            text: Cleaned transcript text potentially containing markers

        Returns:
            Tuple of:
              - cleaned text with markers removed (correction text kept)
              - list of uncertain items, each with "text" and "position" keys
        """
        uncertain_items = []
        offset_adjustment = 0

        for match in _UNCERTAIN_MARKER_RE.finditer(text):
            corrected_text = match.group(1)
            # Position in the final (marker-free) text
            original_start = match.start() - offset_adjustment
            uncertain_items.append({
                "text": corrected_text,
                "position": original_start,
            })
            # Track how much shorter the output gets as we remove markers
            # [[text??]] -> text (remove 6 chars: [[ and ??]])
            offset_adjustment += 6

        # Remove all markers, keeping the corrected text inside
        cleaned = _UNCERTAIN_MARKER_RE.sub(r"\1", text)

        if uncertain_items:
            logger.info(f"Found {len(uncertain_items)} uncertain corrections")

        return cleaned, uncertain_items

    def cleanup_transcript(self, transcript: str, chunk_size: int = 3000) -> Tuple[str, List[Dict]]:
        """
        Clean up transcript using LLM.

        Processes in chunks to stay within context limits.
        Fixes domain terms, removes fillers, corrects errors.
        Returns uncertain corrections separately for review.

        Args:
            transcript: Raw transcript to clean
            chunk_size: Characters per chunk for processing

        Returns:
            Tuple of (cleaned transcript, list of uncertain corrections)
        """
        if not transcript.strip():
            return transcript, []

        logger.info("Starting LLM transcript cleanup")

        # Split by lines to avoid breaking mid-sentence
        lines = transcript.split("\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for line in lines:
            if current_length + len(line) > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = len(line)
            else:
                current_chunk.append(line)
                current_length += len(line) + 1

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        logger.info(f"Processing {len(chunks)} chunks for cleanup")

        cleaned_chunks = []
        all_uncertain = []
        running_offset = 0

        for i, chunk in enumerate(chunks):
            try:
                result = self._generate_sync(CLEANUP_PROMPT.format(chunk=chunk))
                cleaned_result = result.strip()

                # Strip common chatty prefixes
                prefixes = [
                    "Here is the cleaned transcript:",
                    "Here is the cleaned transcript",
                    "Cleaned transcript:",
                    "Sure, here is the cleaned transcript:",
                ]
                for prefix in prefixes:
                    if cleaned_result.lower().startswith(prefix.lower()):
                        cleaned_result = cleaned_result[len(prefix):].strip()
                        break

                # Validate LLM response is a real cleaned transcript
                if self._is_valid_cleanup_response(chunk, cleaned_result):
                    # Extract uncertainty markers before adding to output
                    cleaned_result, uncertain = self.extract_uncertain_corrections(cleaned_result)
                    # Adjust positions to be relative to full transcript
                    for item in uncertain:
                        item["position"] += running_offset
                    all_uncertain.extend(uncertain)
                    cleaned_chunks.append(cleaned_result)
                    logger.debug(f"Cleaned chunk {i+1}/{len(chunks)}")
                else:
                    logger.warning(f"Invalid LLM response for chunk {i+1}, keeping original. Response preview: {cleaned_result[:500]}...")
                    cleaned_chunks.append(chunk)  # Keep original
            except Exception as e:
                logger.warning(f"Cleanup failed for chunk {i+1}: {e}")
                cleaned_chunks.append(chunk)  # Keep original on failure

            running_offset += len(cleaned_chunks[-1]) + 1  # +1 for newline

        logger.info("Transcript cleanup complete")
        return "\n".join(cleaned_chunks), all_uncertain

    def extract_keywords(self, transcript: str) -> List[str]:
        """
        Extract relevant keywords from transcript.

        Args:
            transcript: Full transcript text

        Returns:
            List of extracted keywords
        """
        logger.info("Extracting keywords")

        # Use first 6000 chars to fit in context
        truncated = transcript[:6000]

        try:
            result = self._generate_sync(KEYWORD_PROMPT.format(transcript=truncated))

            # Parse JSON array from response
            # Handle cases where LLM adds extra text
            result = result.strip()
            if result.startswith("["):
                end = result.rfind("]") + 1
                result = result[:end]
            else:
                # Try to find JSON array in response
                start = result.find("[")
                end = result.rfind("]") + 1
                if start != -1 and end > start:
                    result = result[start:end]

            keywords = json.loads(result)
            logger.info(f"Extracted {len(keywords)} keywords")
            return keywords

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse keywords JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []

    def detect_sections(self, transcript: str) -> List[dict]:
        """
        Detect structural sections/chapters in the podcast.

        Args:
            transcript: Full transcript text

        Returns:
            List of section dicts with name, start_time, description
        """
        logger.info("Detecting sections")

        # Use first and last parts plus middle sample for context
        if len(transcript) > 10000:
            sample = (
                transcript[:4000]
                + "\n\n[...middle section...]\n\n"
                + transcript[-4000:]
            )
        else:
            sample = transcript

        try:
            result = self._generate_sync(SECTION_PROMPT.format(transcript=sample))

            # Parse JSON
            result = result.strip()
            start = result.find("[")
            end = result.rfind("]") + 1
            if start != -1 and end > start:
                result = result[start:end]

            sections = json.loads(result)
            logger.info(f"Detected {len(sections)} sections")
            return sections

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse sections JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Section detection failed: {e}")
            return []

    def summarize(self, transcript: str) -> str:
        """
        Generate a concise episode summary.

        Args:
            transcript: Full transcript text

        Returns:
            Summary string (3-5 sentences), or empty string on failure
        """
        logger.info("Generating summary")

        # Use first and last parts for context
        if len(transcript) > 12000:
            sample = (
                transcript[:5000]
                + "\n\n[...]\n\n"
                + transcript[-5000:]
            )
        else:
            sample = transcript

        try:
            result = self._generate_sync(SUMMARY_PROMPT.format(transcript=sample))
            summary = result.strip()

            # Strip common LLM preambles
            preambles = [
                "Here is the summary:",
                "Here's the summary:",
                "Summary:",
            ]
            for preamble in preambles:
                if summary.lower().startswith(preamble.lower()):
                    summary = summary[len(preamble):].strip()
                    break

            # Basic validation: should be at least a sentence
            if len(summary) < 20:
                logger.warning(f"Summary too short ({len(summary)} chars), discarding")
                return ""

            logger.info(f"Generated summary ({len(summary)} chars)")
            return summary

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return ""

    def extract_key_points(self, transcript: str) -> List[str]:
        """
        Extract key takeaways/insights from the transcript.

        Args:
            transcript: Full transcript text

        Returns:
            List of key point strings
        """
        logger.info("Extracting key points")

        # Use first and last parts for context
        if len(transcript) > 12000:
            sample = (
                transcript[:5000]
                + "\n\n[...]\n\n"
                + transcript[-5000:]
            )
        else:
            sample = transcript

        try:
            result = self._generate_sync(KEY_POINTS_PROMPT.format(transcript=sample))

            # Parse JSON array from response
            result = result.strip()
            start = result.find("[")
            end = result.rfind("]") + 1
            if start != -1 and end > start:
                result = result[start:end]

            points = json.loads(result)
            logger.info(f"Extracted {len(points)} key points")
            return points

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse key points JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Key point extraction failed: {e}")
            return []

    def enhance(
        self,
        transcript: str,
        cleanup: bool = True,
        extract_kw: bool = True,
        detect_sect: bool = True,
        gen_summary: bool = True,
        extract_points: bool = True,
    ) -> dict:
        """
        Run full enhancement pipeline.

        Args:
            transcript: Raw transcript
            cleanup: Whether to run LLM cleanup
            extract_kw: Whether to extract keywords
            detect_sect: Whether to detect sections
            gen_summary: Whether to generate a summary
            extract_points: Whether to extract key points

        Returns:
            Dict with cleaned_transcript, keywords, sections, summary,
            key_points, uncertain_corrections
        """
        result = {
            "cleaned_transcript": transcript,
            "keywords": [],
            "sections": [],
            "summary": "",
            "key_points": [],
            "uncertain_corrections": [],
        }

        if cleanup:
            cleaned, uncertain = self.cleanup_transcript(transcript)
            result["cleaned_transcript"] = cleaned
            result["uncertain_corrections"] = uncertain

        working_transcript = result["cleaned_transcript"]

        if extract_kw:
            result["keywords"] = self.extract_keywords(working_transcript)

        if detect_sect:
            result["sections"] = self.detect_sections(working_transcript)

        if gen_summary:
            result["summary"] = self.summarize(working_transcript)

        if extract_points:
            result["key_points"] = self.extract_key_points(working_transcript)

        return result
