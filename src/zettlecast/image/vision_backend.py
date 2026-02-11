"""
Vision Backend for Image Analysis

Integrates with Qwen2.5-VL via Ollama for image understanding, OCR, and concept extraction.
"""

import base64
import json
import logging
from pathlib import Path
from typing import List

import httpx

from .models import VisionExtraction

logger = logging.getLogger(__name__)


class VisionModelNotFoundError(Exception):
    """Raised when the vision model is not available in Ollama."""
    pass


class VisionBackend:
    """
    Interface to Qwen2.5-VL via Ollama for vision tasks.

    Handles image encoding, API communication, and result parsing.
    """

    # Prompts optimized for Qwen2.5-VL
    DESCRIPTION_PROMPT = (
        "Describe this image in 2-3 detailed sentences. "
        "Focus on the main subject, setting, and notable details. "
        "Be specific and factual."
    )

    OCR_PROMPT = (
        "Extract ALL visible text from this image. "
        "Include text from signs, documents, UI elements, captions, etc. "
        "Preserve formatting where relevant. "
        "If no text is visible, respond with exactly: 'No text detected.'"
    )

    CONCEPTS_PROMPT = (
        "List 5-7 key objects, concepts, or themes visible in this image. "
        "Return ONLY a JSON array of lowercase strings, nothing else. "
        "Example: [\"laptop\", \"code editor\", \"dark theme\", \"programming\", \"python\"]\n\n"
        "JSON array:"
    )

    def __init__(self, model: str, base_url: str):
        """
        Initialize vision backend.

        Args:
            model: Ollama model name (e.g., "qwen2.5-vl:7b")
            base_url: Ollama API base URL (e.g., "http://localhost:11434")
        """
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)  # Vision models are slower

    def _encode_image(self, image_path: Path) -> str:
        """
        Encode image to base64 for Ollama API.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode("utf-8")

    def _call_vision_api(self, image_data: str, prompt: str) -> str:
        """
        Call Ollama vision endpoint with image + prompt.

        Args:
            image_data: Base64 encoded image
            prompt: Text prompt for vision model

        Returns:
            Model response text

        Raises:
            VisionModelNotFoundError: If model not available
            httpx.HTTPStatusError: On API errors
        """
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["response"].strip()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise VisionModelNotFoundError(
                    f"Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}"
                )
            logger.error(f"Vision API error: {e}")
            raise

        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            raise

    def _parse_concepts(self, response: str) -> List[str]:
        """
        Parse JSON array of concepts from model response.

        Args:
            response: Model response (should be JSON array)

        Returns:
            List of concept strings
        """
        try:
            # Try to find JSON array in response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                concepts = json.loads(json_str)
                if isinstance(concepts, list):
                    return [str(c).lower().strip() for c in concepts if c]
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse concepts JSON: {e}")

        # Fallback: split by commas/newlines
        return [
            c.strip().lower()
            for c in response.replace("[", "").replace("]", "").replace('"', "").split(",")
            if c.strip()
        ][:7]

    def analyze_image(self, image_path: Path) -> VisionExtraction:
        """
        Run complete vision analysis on image.

        Performs three API calls:
        1. Visual description
        2. OCR text extraction
        3. Concept/object tagging

        Args:
            image_path: Path to image file

        Returns:
            VisionExtraction with all analysis results

        Raises:
            VisionModelNotFoundError: If model not available
            FileNotFoundError: If image file doesn't exist
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Analyzing image with {self.model}: {image_path.name}")

        # Encode image once, use for all API calls
        image_data = self._encode_image(image_path)

        # 1. Get visual description
        logger.debug("Extracting visual description...")
        description = self._call_vision_api(image_data, self.DESCRIPTION_PROMPT)

        # 2. Extract OCR text
        logger.debug("Extracting text (OCR)...")
        detected_text = self._call_vision_api(image_data, self.OCR_PROMPT)

        # 3. Extract concepts/tags
        logger.debug("Extracting concepts...")
        concepts_response = self._call_vision_api(image_data, self.CONCEPTS_PROMPT)
        concepts = self._parse_concepts(concepts_response)

        return VisionExtraction(
            description=description,
            detected_text=detected_text,
            concepts=concepts,
        )

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()
