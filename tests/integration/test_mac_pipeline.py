"""
Integration tests for Mac ASR pipeline.

These tests require:
- macOS with Apple Silicon
- parakeet-mlx installed
- pyannote.audio installed with HF token

Run with: pytest tests/integration/test_mac_pipeline.py -v
"""

import os
import pytest
from pathlib import Path


# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    os.uname().sysname != "Darwin",
    reason="Mac integration tests only run on macOS"
)


@pytest.fixture
def sample_audio_path():
    """Path to sample audio file for testing."""
    # Check for test fixture
    fixture_path = Path(__file__).parent.parent / "fixtures" / "test_audio.wav"
    
    if not fixture_path.exists():
        pytest.skip(
            f"Sample audio not found at {fixture_path}. "
            "Create with: ffmpeg -f lavfi -i 'sine=frequency=440:duration=5' tests/fixtures/test_audio.wav"
        )
    
    return fixture_path


@pytest.fixture
def hf_token():
    """HuggingFace token for pyannote models."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN environment variable not set")
    return token


class TestMacPipelineIntegration:
    """Integration tests for the full Mac transcription pipeline."""

    def test_factory_creates_mac_transcriber_on_mac(self):
        """Factory should create MacTranscriber on macOS Apple Silicon."""
        from zettlecast.podcast.transcriber_factory import TranscriberFactory
        
        transcriber = TranscriberFactory.create()
        
        caps = transcriber.get_capabilities()
        # On Mac, should be either parakeet-mlx or whisper fallback
        assert caps.platform in ("darwin", "any")

    def test_mac_transcriber_warmup(self, hf_token):
        """Should be able to warmup (preload) models."""
        try:
            from zettlecast.podcast.mac_transcriber import MacTranscriber
            from zettlecast.podcast.base_transcriber import TranscriberConfig
        except ImportError:
            pytest.skip("parakeet-mlx not installed")
        
        config = TranscriberConfig(
            hf_token=hf_token,
            enable_diarization=False,  # Skip diarization for faster test
        )
        
        transcriber = MacTranscriber(config)
        
        # Warmup should not raise
        transcriber.warmup()

    def test_transcribe_short_audio(self, sample_audio_path, hf_token):
        """Should transcribe a short audio file."""
        try:
            from zettlecast.podcast.mac_transcriber import MacTranscriber
            from zettlecast.podcast.base_transcriber import TranscriberConfig
        except ImportError:
            pytest.skip("parakeet-mlx not installed")
        
        config = TranscriberConfig(
            hf_token=hf_token,
            enable_diarization=False,
        )
        
        transcriber = MacTranscriber(config)
        result = transcriber.transcribe(sample_audio_path)
        
        assert result is not None
        assert result.duration_seconds > 0
        assert result.processing_time_seconds > 0
        # For a sine wave, there may be no speech detected
        assert isinstance(result.full_text, str)

    def test_transcribe_with_diarization(self, sample_audio_path, hf_token):
        """Should transcribe with speaker diarization."""
        try:
            from zettlecast.podcast.mac_transcriber import MacTranscriber
            from zettlecast.podcast.base_transcriber import TranscriberConfig
        except ImportError:
            pytest.skip("parakeet-mlx not installed")
        
        config = TranscriberConfig(
            hf_token=hf_token,
            enable_diarization=True,
        )
        
        transcriber = MacTranscriber(config)
        result = transcriber.transcribe(sample_audio_path)
        
        assert result is not None
        assert result.speakers_detected >= 0  # May be 0 for non-speech audio
