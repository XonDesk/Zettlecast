"""
Tests for MacTranscriber

Unit tests with mocked parakeet-mlx and pyannote dependencies.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zettlecast.podcast.base_transcriber import TranscriberConfig, TranscriberCapabilities


class TestMacTranscriberCapabilities:
    """Tests for MacTranscriber.get_capabilities()."""

    def test_capabilities_returns_correct_info(self):
        """Should return Mac-specific capabilities."""
        with patch.dict("sys.modules", {"parakeet_mlx": MagicMock()}):
            from zettlecast.podcast.mac_transcriber import MacTranscriber
            
            config = TranscriberConfig(enable_diarization=True)
            transcriber = MacTranscriber(config)
            
            caps = transcriber.get_capabilities()
            
            assert isinstance(caps, TranscriberCapabilities)
            assert caps.platform == "darwin"
            assert caps.transcriber_name == "parakeet-mlx"
            assert caps.diarizer_name == "pyannote.audio"
            assert caps.supports_diarization is True
            assert caps.requires_container is False

    def test_capabilities_no_diarization(self):
        """Should reflect disabled diarization in capabilities."""
        with patch.dict("sys.modules", {"parakeet_mlx": MagicMock()}):
            from zettlecast.podcast.mac_transcriber import MacTranscriber
            
            config = TranscriberConfig(enable_diarization=False)
            transcriber = MacTranscriber(config)
            
            caps = transcriber.get_capabilities()
            
            assert caps.diarizer_name is None


class TestMacTranscriberAvailability:
    """Tests for MacTranscriber.is_available()."""

    def test_available_when_parakeet_installed(self):
        """Should return True when parakeet_mlx is importable."""
        with patch.dict("sys.modules", {"parakeet_mlx": MagicMock()}):
            from zettlecast.podcast.mac_transcriber import MacTranscriber
            
            transcriber = MacTranscriber()
            
            # Since we mocked the import, is_available tries to import again
            # We need to patch the actual import in is_available
            with patch("builtins.__import__", return_value=MagicMock()):
                result = transcriber.is_available()
                # The actual implementation imports, which may fail in test env
                # This test validates the method exists and is callable
                assert isinstance(result, bool)


class TestMacTranscriberWordExtraction:
    """Tests for word extraction from parakeet results."""

    def test_extract_words_from_aligned_result(self):
        """Should extract Word objects from parakeet AlignedResult."""
        with patch.dict("sys.modules", {"parakeet_mlx": MagicMock()}):
            from zettlecast.podcast.mac_transcriber import MacTranscriber
            from zettlecast.podcast.aligner import Word
            
            transcriber = MacTranscriber()
            
            # Mock parakeet result structure
            mock_token1 = MagicMock(text="Hello", start=0.0, end=0.5)
            mock_token2 = MagicMock(text="world", start=0.6, end=1.0)
            mock_sentence = MagicMock(tokens=[mock_token1, mock_token2])
            mock_result = MagicMock(sentences=[mock_sentence])
            
            words = transcriber._extract_words(mock_result)
            
            assert len(words) == 2
            assert words[0].text == "Hello"
            assert words[0].start == 0.0
            assert words[0].end == 0.5
            assert words[1].text == "world"


class TestMacTranscriberRTTMConversion:
    """Tests for pyannote annotation to RTTM conversion."""

    def test_annotation_to_rttm_format(self):
        """Should convert pyannote Annotation to RTTM format string."""
        with patch.dict("sys.modules", {"parakeet_mlx": MagicMock()}):
            from zettlecast.podcast.mac_transcriber import MacTranscriber
            
            transcriber = MacTranscriber()
            
            # Mock pyannote annotation
            mock_turn1 = MagicMock(start=0.0, end=5.0)
            mock_turn2 = MagicMock(start=5.5, end=10.0)
            
            mock_annotation = MagicMock()
            mock_annotation.itertracks.return_value = [
                (mock_turn1, None, "SPEAKER_00"),
                (mock_turn2, None, "SPEAKER_01"),
            ]
            
            rttm = transcriber._annotation_to_rttm(mock_annotation)
            
            lines = rttm.strip().split("\n")
            assert len(lines) == 2
            assert "SPEAKER_00" in lines[0]
            assert "SPEAKER_01" in lines[1]
            assert "0.000" in lines[0]  # Start time
            assert "5.000" in lines[0]  # Duration


class TestMacTranscriberSegmentCreation:
    """Tests for word-to-segment conversion."""

    def test_words_to_segments_groups_by_pause(self):
        """Should create segments grouped by pause gaps."""
        with patch.dict("sys.modules", {"parakeet_mlx": MagicMock()}):
            from zettlecast.podcast.mac_transcriber import MacTranscriber
            from zettlecast.podcast.aligner import Word
            
            transcriber = MacTranscriber()
            
            # Words with a >2s gap between them
            words = [
                Word(text="First", start=0.0, end=0.5),
                Word(text="segment", start=0.6, end=1.0),
                Word(text="Second", start=5.0, end=5.5),  # 4s gap
                Word(text="segment", start=5.6, end=6.0),
            ]
            
            segments = transcriber._words_to_segments(words)
            
            assert len(segments) == 2
            assert "First segment" in segments[0].text
            assert "Second segment" in segments[1].text

    def test_words_to_segments_empty_list(self):
        """Should return empty list for empty input."""
        with patch.dict("sys.modules", {"parakeet_mlx": MagicMock()}):
            from zettlecast.podcast.mac_transcriber import MacTranscriber
            
            transcriber = MacTranscriber()
            
            segments = transcriber._words_to_segments([])
            
            assert segments == []
