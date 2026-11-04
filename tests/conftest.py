"""Zettlecast Test Configuration"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def _no_llama_server(monkeypatch):
    """Never start a real llama-server from a test (tests that need it set the backend)."""
    from zettlecast.config import settings

    monkeypatch.setattr(settings, "llm_backend", "ollama")
    # the machine's .env may say primary or worker (deploy/); tests that need a role set it themselves
    monkeypatch.setattr(settings, "role", "standalone")
