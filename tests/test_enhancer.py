"""
Tests for Transcript Enhancer

Unit tests for cleanup prompts, uncertainty marker parsing, and
enhancement pipeline structure. Does NOT require Ollama running -
LLM calls are mocked.
"""

import pytest
from unittest.mock import patch, MagicMock

from zettlecast.podcast.enhancer import (
    TranscriptEnhancer,
    CLEANUP_PROMPT,
    KEYWORD_PROMPT,
    SUMMARY_PROMPT,
    KEY_POINTS_PROMPT,
    _UNCERTAIN_MARKER_RE,
)


class TestCleanupPrompt:
    """Tests for the cleanup prompt content."""

    def test_prompt_is_general_not_running_specific(self):
        """Prompt should be general, not narrowly focused on running."""
        prompt_lower = CLEANUP_PROMPT.lower()
        assert "running and coaching" not in prompt_lower
        assert "podcast transcript" in prompt_lower

    def test_prompt_mentions_domain_corrections(self):
        """Prompt should instruct the LLM to fix domain terms."""
        assert "exercise physiology" in CLEANUP_PROMPT
        assert "VO2 max" in CLEANUP_PROMPT

    def test_prompt_has_uncertainty_instructions(self):
        """Prompt should instruct LLM to mark uncertain corrections."""
        assert "[[" in CLEANUP_PROMPT
        assert "??]]" in CLEANUP_PROMPT
        assert "80%" in CLEANUP_PROMPT


class TestExtractUncertainCorrections:
    """Tests for uncertainty marker parsing."""

    def test_no_markers_returns_unchanged(self):
        """Text without markers should be returned as-is."""
        text = "This is normal text with no markers."
        cleaned, items = TranscriptEnhancer.extract_uncertain_corrections(text)
        assert cleaned == text
        assert items == []

    def test_single_marker_extracted(self):
        """Should extract a single [[text??]] marker."""
        text = "He studied [[exercise physiology??]] at the university."
        cleaned, items = TranscriptEnhancer.extract_uncertain_corrections(text)
        assert cleaned == "He studied exercise physiology at the university."
        assert len(items) == 1
        assert items[0]["text"] == "exercise physiology"

    def test_multiple_markers_extracted(self):
        """Should extract multiple markers."""
        text = "The [[VO2 max??]] and [[lactate threshold??]] were measured."
        cleaned, items = TranscriptEnhancer.extract_uncertain_corrections(text)
        assert cleaned == "The VO2 max and lactate threshold were measured."
        assert len(items) == 2
        assert items[0]["text"] == "VO2 max"
        assert items[1]["text"] == "lactate threshold"

    def test_position_tracking(self):
        """Positions should reflect the marker-free text."""
        text = "Start [[word??]] end."
        cleaned, items = TranscriptEnhancer.extract_uncertain_corrections(text)
        assert cleaned == "Start word end."
        assert items[0]["position"] == 6  # "Start " is 6 chars

    def test_empty_text(self):
        """Should handle empty input."""
        cleaned, items = TranscriptEnhancer.extract_uncertain_corrections("")
        assert cleaned == ""
        assert items == []

    def test_marker_regex_pattern(self):
        """Regex should match [[text??]] but not other brackets."""
        assert _UNCERTAIN_MARKER_RE.search("[[hello??]]")
        assert not _UNCERTAIN_MARKER_RE.search("[hello]")
        assert not _UNCERTAIN_MARKER_RE.search("[[hello]]")
        assert not _UNCERTAIN_MARKER_RE.search("hello??")


class TestValidCleanupResponse:
    """Tests for LLM response validation."""

    @pytest.fixture
    def enhancer(self):
        return TranscriptEnhancer()

    def test_rejects_short_response(self, enhancer):
        """Should reject responses much shorter than original."""
        original = "A" * 100
        response = "Short"
        assert enhancer._is_valid_cleanup_response(original, response) is False

    def test_accepts_similar_length_response(self, enhancer):
        """Should accept responses of similar length to original."""
        original = "Hello world this is a test transcript segment."
        response = "Hello world this is a test transcript segment."
        assert enhancer._is_valid_cleanup_response(original, response) is True

    def test_rejects_placeholder_responses(self, enhancer):
        """Should reject common LLM placeholder patterns."""
        original = "Some transcript text here that is long enough."
        placeholders = [
            "I'd be happy to help clean up the transcript",
            "Please provide the transcript",
            "Here is the cleaned transcript: some text here yes",
        ]
        for placeholder in placeholders:
            assert enhancer._is_valid_cleanup_response(original, placeholder) is False


class TestEnhancePipeline:
    """Tests for the full enhance() pipeline."""

    @pytest.fixture
    def enhancer(self):
        return TranscriptEnhancer()

    def test_enhance_returns_all_keys(self, enhancer):
        """Result dict should contain all expected keys."""
        with patch.object(enhancer, "_generate_sync") as mock_gen:
            mock_gen.return_value = "cleaned text here that is long enough to pass validation check yes"
            result = enhancer.enhance(
                "test transcript",
                cleanup=False,
                extract_kw=False,
                detect_sect=False,
                gen_summary=False,
                extract_points=False,
            )
        expected_keys = {
            "cleaned_transcript",
            "keywords",
            "sections",
            "summary",
            "key_points",
            "uncertain_corrections",
        }
        assert set(result.keys()) == expected_keys

    def test_enhance_without_cleanup_keeps_original(self, enhancer):
        """With cleanup=False, cleaned_transcript should be the original."""
        result = enhancer.enhance(
            "original text",
            cleanup=False,
            extract_kw=False,
            detect_sect=False,
            gen_summary=False,
            extract_points=False,
        )
        assert result["cleaned_transcript"] == "original text"

    def test_enhance_calls_all_steps(self, enhancer):
        """With all steps enabled, all methods should be called."""
        with patch.object(enhancer, "cleanup_transcript", return_value=("cleaned", [])) as mock_cleanup, \
             patch.object(enhancer, "extract_keywords", return_value=["kw"]) as mock_kw, \
             patch.object(enhancer, "detect_sections", return_value=[]) as mock_sect, \
             patch.object(enhancer, "summarize", return_value="summary") as mock_sum, \
             patch.object(enhancer, "extract_key_points", return_value=["point"]) as mock_pts:
            result = enhancer.enhance("test")

            mock_cleanup.assert_called_once()
            mock_kw.assert_called_once()
            mock_sect.assert_called_once()
            mock_sum.assert_called_once()
            mock_pts.assert_called_once()

            assert result["cleaned_transcript"] == "cleaned"
            assert result["keywords"] == ["kw"]
            assert result["summary"] == "summary"
            assert result["key_points"] == ["point"]


class TestChunking:
    """Tests for transcript chunking in cleanup."""

    @pytest.fixture
    def enhancer(self):
        return TranscriptEnhancer()

    def test_single_chunk_for_short_transcript(self, enhancer):
        """Short transcripts should produce a single chunk."""
        transcript = "Short text."
        with patch.object(enhancer, "_generate_sync") as mock_gen:
            # Return text that passes validation
            mock_gen.return_value = "Short text."
            enhancer.cleanup_transcript(transcript, chunk_size=3000)
            assert mock_gen.call_count == 1

    def test_multiple_chunks_for_long_transcript(self, enhancer):
        """Long transcripts should be split into multiple chunks."""
        # Create a transcript with multiple lines exceeding chunk_size
        transcript = "\n".join([f"Line {i} " + "x" * 50 for i in range(100)])
        with patch.object(enhancer, "_generate_sync") as mock_gen:
            mock_gen.return_value = "x" * 200  # Long enough to pass validation
            enhancer.cleanup_transcript(transcript, chunk_size=500)
            assert mock_gen.call_count > 1
