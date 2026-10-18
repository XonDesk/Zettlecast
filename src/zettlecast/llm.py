"""
Ollama client shared by every LLM step (transcript enhancement, tagging,
image analysis).

All calls go through /api/chat with thinking disabled and an explicit context
window. Pass a JSON schema to get structured output back as a dict.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Any, Optional

import httpx

from .config import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when Ollama is unreachable or returns an unusable response."""


def chat(
    prompt: str,
    schema: Optional[dict] = None,
    images: Optional[list[Path]] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.2,
) -> Any:
    """
    Run one chat turn against Ollama.

    Args:
        prompt: User message
        schema: JSON schema; when given, the response is parsed and returned as a dict
        images: Image files to attach (vision)
        model: Override settings.ollama_model
        system: Optional system message
        temperature: Sampling temperature

    Returns:
        Response text, or the parsed JSON object when schema is given
    """
    user_msg: dict = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = [base64.b64encode(Path(p).read_bytes()).decode() for p in images]

    messages = [{"role": "system", "content": system}] if system else []
    messages.append(user_msg)

    body: dict = {
        "model": model or settings.ollama_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"num_ctx": settings.llm_num_ctx, "temperature": temperature},
    }
    if settings.llm_seed is not None:
        body["options"]["seed"] = settings.llm_seed
    if schema:
        body["format"] = schema

    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json=body,
            timeout=settings.llm_timeout_seconds,
        )
    except httpx.ConnectError as e:
        raise LLMError(f"Cannot connect to Ollama at {settings.ollama_base_url}") from e

    if response.status_code == 404:
        raise LLMError(f"Model '{body['model']}' not found. Run: ollama pull {body['model']}")
    response.raise_for_status()

    content = response.json()["message"]["content"].strip()
    if not schema:
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMError(f"Model returned invalid JSON: {content[:200]}") from e


def unload(model: Optional[str] = None) -> None:
    """Ask Ollama to free the model's VRAM now (before GPU-heavy ASR work)."""
    try:
        httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": model or settings.ollama_model, "keep_alive": 0},
            timeout=30,
        )
    except httpx.HTTPError as e:
        logger.debug(f"Ollama unload skipped: {e}")
