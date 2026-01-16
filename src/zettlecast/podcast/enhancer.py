"""
Transcript Enhancement Module

LLM-based cleanup, keyword extraction, and section detection using Ollama.
"""

import json
import logging
from typing import List, Optional

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


CLEANUP_PROMPT = """You are cleaning up an automated podcast transcript about running and coaching.

Fix these issues while preserving timestamps [X.Xs]:
1. Running terms: "tempo run" not "temple run", "fartlek" not "far fleck", "cadence" not "cadets"
2. Remove filler words: um, uh, like, you know, I mean (unless meaningful)
3. Fix obvious transcription errors from context
4. Keep speaker labels (Speaker 1:, Speaker 2:) intact
5. Do NOT change meaning or add content

Return ONLY the cleaned transcript, preserving all timestamps.

Transcript chunk:
{chunk}"""

KEYWORD_PROMPT = """Extract 5-10 keywords from this running/coaching podcast transcript.
Focus on: training concepts, race types, injury prevention, gear, nutrition, coaching philosophy.
Return ONLY a JSON array of strings, nothing else.

Example: ["marathon training", "tempo runs", "injury prevention"]

Transcript (first 6000 characters):
{transcript}"""

SECTION_PROMPT = """Identify chapters/sections in this podcast transcript.
Common sections: introduction, main topic discussion, sponsor reads, Q&A, conclusion/outro.

Return ONLY a JSON array of objects with "name", "start_time", and "description" fields.
Times should be in seconds (numbers, not strings).

Example: [{"name": "Introduction", "start_time": 0.0, "description": "Hosts introduce the topic"}]

Transcript:
{transcript}"""


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

    def cleanup_transcript(self, transcript: str, chunk_size: int = 3000) -> str:
        """
        Clean up transcript using LLM.

        Processes in chunks to stay within context limits.
        Fixes domain terms, removes fillers, corrects errors.

        Args:
            transcript: Raw transcript to clean
            chunk_size: Characters per chunk for processing

        Returns:
            Cleaned transcript
        """
        if not transcript.strip():
            return transcript

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
        for i, chunk in enumerate(chunks):
            try:
                result = self._generate_sync(CLEANUP_PROMPT.format(chunk=chunk))
                cleaned_chunks.append(result.strip())
                logger.debug(f"Cleaned chunk {i+1}/{len(chunks)}")
            except Exception as e:
                logger.warning(f"Cleanup failed for chunk {i+1}: {e}")
                cleaned_chunks.append(chunk)  # Keep original on failure

        logger.info("Transcript cleanup complete")
        return "\n".join(cleaned_chunks)

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

    def enhance(
        self,
        transcript: str,
        cleanup: bool = True,
        extract_kw: bool = True,
        detect_sect: bool = True,
    ) -> dict:
        """
        Run full enhancement pipeline.

        Args:
            transcript: Raw transcript
            cleanup: Whether to run LLM cleanup
            extract_kw: Whether to extract keywords
            detect_sect: Whether to detect sections

        Returns:
            Dict with cleaned_transcript, keywords, sections
        """
        result = {
            "cleaned_transcript": transcript,
            "keywords": [],
            "sections": [],
        }

        if cleanup:
            result["cleaned_transcript"] = self.cleanup_transcript(transcript)

        if extract_kw:
            result["keywords"] = self.extract_keywords(
                result["cleaned_transcript"] if cleanup else transcript
            )

        if detect_sect:
            result["sections"] = self.detect_sections(
                result["cleaned_transcript"] if cleanup else transcript
            )

        return result
