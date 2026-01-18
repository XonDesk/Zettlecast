"""
Content Enricher Module

LLM-based tag extraction and summary generation for all note types.
Uses configured LLM provider (Ollama by default) to auto-generate metadata.
"""

import json
import logging
from typing import List, Optional

import httpx

from .config import settings
from .models import NoteModel

logger = logging.getLogger(__name__)


TAG_PROMPT = """Extract exactly {count} keywords/tags from this text that best describe the main topics.
Return ONLY a JSON array of lowercase strings, nothing else.

Example: ["machine learning", "neural networks", "deep learning", "python", "tensorflow"]

Text (first 4000 characters):
{text}"""

SUMMARY_PROMPT = """Write a single, concise sentence (max 150 characters) summarizing the main point of this text.
Return ONLY the summary sentence, nothing else.

Text (first 3000 characters):
{text}"""


class ContentEnricher:
    """
    Enrich notes with LLM-generated tags and summaries.
    """

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
    ):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model

    def _generate_sync(self, prompt: str, timeout: float = 60.0) -> str:
        """Call Ollama generate endpoint synchronously."""
        try:
            with httpx.Client(timeout=timeout) as client:
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
        except httpx.ConnectError:
            logger.warning(f"Cannot connect to Ollama at {self.base_url}")
            return ""
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""

    def extract_tags(self, text: str, count: int = None) -> List[str]:
        """
        Extract keywords/tags from text using LLM.

        Args:
            text: Source text to analyze
            count: Number of tags to extract (default from settings)

        Returns:
            List of lowercase tag strings
        """
        count = count or settings.auto_tag_count
        
        if not text or len(text.strip()) < 50:
            logger.debug("Text too short for tag extraction")
            return []

        logger.info(f"Extracting {count} tags via LLM")
        
        # Truncate to first 4000 chars
        truncated = text[:4000]
        
        try:
            result = self._generate_sync(
                TAG_PROMPT.format(count=count, text=truncated)
            )
            
            if not result:
                return []

            # Parse JSON array from response
            result = result.strip()
            
            # Extract JSON array even if LLM adds extra text
            start = result.find("[")
            end = result.rfind("]") + 1
            
            if start != -1 and end > start:
                result = result[start:end]
                tags = json.loads(result)
                
                # Normalize: lowercase, strip whitespace
                tags = [t.strip().lower() for t in tags if isinstance(t, str)]
                logger.info(f"Extracted tags: {tags}")
                return tags[:count]

            logger.warning("No JSON array found in LLM response")
            return []

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tags JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Tag extraction failed: {e}")
            return []

    def generate_summary(self, text: str) -> Optional[str]:
        """
        Generate a 1-sentence summary of the text.

        Args:
            text: Source text to summarize

        Returns:
            Summary string or None on failure
        """
        if not text or len(text.strip()) < 100:
            return None

        logger.info("Generating summary via LLM")
        
        truncated = text[:3000]
        
        try:
            result = self._generate_sync(
                SUMMARY_PROMPT.format(text=truncated)
            )
            
            if result:
                summary = result.strip()
                # Remove quotes if LLM wrapped in them
                summary = summary.strip('"\'')
                logger.info(f"Generated summary: {summary[:80]}...")
                return summary
            return None

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return None


def enrich_note(note: NoteModel) -> NoteModel:
    """
    Enrich a note with LLM-generated tags and summary.
    
    This is the main entry point called during ingestion.
    Modifies note.metadata.tags and note.metadata.custom["summary"].
    
    Args:
        note: The note to enrich
        
    Returns:
        The enriched note (same object, modified in place)
    """
    if not settings.enable_auto_tagging:
        logger.debug("Auto-tagging disabled, skipping enrichment")
        return note
    
    enricher = ContentEnricher()
    
    # Extract tags if none exist
    if not note.metadata.tags:
        tags = enricher.extract_tags(note.full_text)
        if tags:
            note.metadata.tags = tags
    
    # Generate summary if not present
    if "summary" not in note.metadata.custom:
        summary = enricher.generate_summary(note.full_text)
        if summary:
            note.metadata.custom["summary"] = summary
    
    return note


# Convenience function for standalone use
def extract_tags(text: str, count: int = 5) -> List[str]:
    """Extract tags from text. Convenience wrapper."""
    return ContentEnricher().extract_tags(text, count)


def generate_summary(text: str) -> Optional[str]:
    """Generate summary from text. Convenience wrapper."""
    return ContentEnricher().generate_summary(text)
