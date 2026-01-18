"""
Tests for TranscriberFactory

Unit tests for platform detection and backend selection.
"""

import platform
from unittest.mock import MagicMock, patch

import pytest

from zettlecast.podcast.base_transcriber import TranscriberConfig


class TestDetectPlatform:
    """Tests for platform detection logic."""

    @patch("platform.system")
    @patch("platform.machine")
    def test_detects_darwin_arm64(self, mock_machine, mock_system):
        """Should detect macOS Apple Silicon."""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"
        
        from zettlecast.podcast.transcriber_factory import detect_platform
        
        # Reset cache
        from zettlecast.podcast import transcriber_factory
        transcriber_factory.TranscriberFactory._platform_cache = None
        
        result = detect_platform()
        assert result == "darwin"

    @patch("platform.system")
    @patch("platform.machine")
    def test_detects_darwin_intel_as_cpu(self, mock_machine, mock_system):
        """Should detect macOS Intel as CPU-only."""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "x86_64"
        
        from zettlecast.podcast.transcriber_factory import detect_platform
        from zettlecast.podcast import transcriber_factory
        transcriber_factory.TranscriberFactory._platform_cache = None
        
        result = detect_platform()
        assert result == "cpu"

    @patch("platform.system")
    @patch("torch.cuda.is_available")
    def test_detects_windows_with_cuda(self, mock_cuda, mock_system):
        """Should detect Windows with CUDA GPU."""
        mock_system.return_value = "Windows"
        mock_cuda.return_value = True
        
        with patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3080"):
            from zettlecast.podcast.transcriber_factory import detect_platform
            from zettlecast.podcast import transcriber_factory
            transcriber_factory.TranscriberFactory._platform_cache = None
            
            result = detect_platform()
            assert result == "win32+cuda"

    @patch("platform.system")
    def test_detects_cpu_fallback(self, mock_system):
        """Should fall back to CPU when no GPU available."""
        mock_system.return_value = "Linux"
        
        # Mock torch not having CUDA
        with patch.dict("sys.modules", {"torch": MagicMock(cuda=MagicMock(is_available=lambda: False))}):
            from zettlecast.podcast.transcriber_factory import detect_platform
            from zettlecast.podcast import transcriber_factory
            transcriber_factory.TranscriberFactory._platform_cache = None
            
            result = detect_platform()
            assert result == "cpu"


class TestTranscriberFactory:
    """Tests for TranscriberFactory.create()."""

    def test_create_returns_base_transcriber_instance(self):
        """Factory should return a BaseTranscriber subclass."""
        from zettlecast.podcast.transcriber_factory import TranscriberFactory
        from zettlecast.podcast.base_transcriber import BaseTranscriber
        
        # Will create whisper transcriber as fallback
        transcriber = TranscriberFactory.create()
        
        assert isinstance(transcriber, BaseTranscriber)

    def test_create_with_explicit_whisper_backend(self):
        """Should create WhisperTranscriber when explicitly requested."""
        from zettlecast.podcast.transcriber_factory import TranscriberFactory
        from zettlecast.podcast.whisper_transcriber import WhisperTranscriber
        
        transcriber = TranscriberFactory.create(backend="whisper")
        
        assert isinstance(transcriber, WhisperTranscriber)

    def test_create_respects_config(self):
        """Config should be passed to created transcriber."""
        from zettlecast.podcast.transcriber_factory import TranscriberFactory
        
        config = TranscriberConfig(
            enable_diarization=False,
            chunk_duration_minutes=15,
            language="es",
        )
        
        transcriber = TranscriberFactory.create(config=config, backend="whisper")
        
        assert transcriber.config.enable_diarization is False
        assert transcriber.config.chunk_duration_minutes == 15
        assert transcriber.config.language == "es"

    def test_list_available_backends_returns_list(self):
        """Should return list of backend info dicts."""
        from zettlecast.podcast.transcriber_factory import TranscriberFactory
        
        backends = TranscriberFactory.list_available_backends()
        
        assert isinstance(backends, list)
        assert len(backends) >= 1  # At least whisper should be available
        
        # Each backend should have required keys
        for backend in backends:
            assert "name" in backend
            assert "available" in backend

    def test_whisper_always_available(self):
        """Whisper backend should always be listed as available."""
        from zettlecast.podcast.transcriber_factory import TranscriberFactory
        
        backends = TranscriberFactory.list_available_backends()
        
        whisper_backend = next(
            (b for b in backends if b["name"] == "whisper"),
            None
        )
        
        assert whisper_backend is not None
        assert whisper_backend["available"] is True


class TestTranscriberConfig:
    """Tests for TranscriberConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = TranscriberConfig()
        
        assert config.device == "auto"
        assert config.enable_diarization is True
        assert config.chunk_duration_minutes == 10
        assert config.language == "en"
        assert config.beam_size == 5

    def test_custom_values(self):
        """Config should accept custom values."""
        config = TranscriberConfig(
            device="cuda",
            enable_diarization=False,
            chunk_duration_minutes=20,
            hf_token="test_token",
        )
        
        assert config.device == "cuda"
        assert config.enable_diarization is False
        assert config.chunk_duration_minutes == 20
        assert config.hf_token == "test_token"
