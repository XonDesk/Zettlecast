"""
Unit tests for the enricher module.
Tests LLM-based tag extraction and summary generation.
"""

import pytest
from unittest.mock import MagicMock, patch

from zettlecast.enricher import (
    extract_tags,
    generate_summary,
    enrich_note,
    ContentEnricher,
)
from zettlecast.models import NoteModel, NoteMetadata


class TestExtractTags:
    """Tests for tag extraction functions."""
    
    @patch("zettlecast.enricher.ContentEnricher._generate_sync")
    def test_extract_tags_parses_json(self, mock_generate):
        """Should parse JSON array from LLM response."""
        mock_generate.return_value = '["python", "machine learning", "ai"]'
        
        enricher = ContentEnricher()
        # Text must be > 50 chars to pass the length check
        tags = enricher.extract_tags("Some text about Python and AI " * 5)
        
        assert tags == ["python", "machine learning", "ai"]
    
    @patch("zettlecast.enricher.ContentEnricher._generate_sync")
    def test_extract_tags_handles_extra_text(self, mock_generate):
        """Should extract JSON even with surrounding text."""
        mock_generate.return_value = 'Here are the tags: ["python", "ai", "ml"] hope this helps!'
        
        enricher = ContentEnricher()
        tags = enricher.extract_tags("Some text about programming and data science " * 3)
        
        assert tags == ["python", "ai", "ml"]
    
    @patch("zettlecast.enricher.ContentEnricher._generate_sync")
    def test_extract_tags_normalizes_case(self, mock_generate):
        """Tags should be lowercased."""
        mock_generate.return_value = '["Python", "MACHINE LEARNING", "Ai"]'
        
        enricher = ContentEnricher()
        tags = enricher.extract_tags("Some text about coding and technology " * 3)
        
        assert tags == ["python", "machine learning", "ai"]
    
    @patch("zettlecast.enricher.ContentEnricher._generate_sync")
    def test_extract_tags_limits_count(self, mock_generate):
        """Should limit to requested count."""
        mock_generate.return_value = '["a", "b", "c", "d", "e", "f"]'
        
        enricher = ContentEnricher()
        tags = enricher.extract_tags("Some text about many different topics " * 3, count=3)
        
        assert len(tags) == 3
    
    def test_extract_tags_empty_text(self):
        """Should return empty list for short text."""
        enricher = ContentEnricher()
        tags = enricher.extract_tags("")
        
        assert tags == []
    
    @patch("zettlecast.enricher.ContentEnricher._generate_sync")
    def test_extract_tags_handles_invalid_json(self, mock_generate):
        """Should return empty list on JSON parse error."""
        mock_generate.return_value = "not valid json at all"
        
        enricher = ContentEnricher()
        tags = enricher.extract_tags("Some text about stuff")
        
        assert tags == []


class TestGenerateSummary:
    """Tests for summary generation."""
    
    @patch("zettlecast.enricher.ContentEnricher._generate_sync")
    def test_generate_summary_basic(self, mock_generate):
        """Should return cleaned summary."""
        mock_generate.return_value = '"This is a summary of the article."'
        
        enricher = ContentEnricher()
        summary = enricher.generate_summary("Long article text " * 100)
        
        assert summary == "This is a summary of the article."
    
    @patch("zettlecast.enricher.ContentEnricher._generate_sync")
    def test_generate_summary_strips_quotes(self, mock_generate):
        """Should strip surrounding quotes."""
        mock_generate.return_value = "'Quoted summary'"
        
        enricher = ContentEnricher()
        summary = enricher.generate_summary("Long text " * 100)
        
        assert summary == "Quoted summary"
    
    def test_generate_summary_short_text(self):
        """Should return None for very short text."""
        enricher = ContentEnricher()
        summary = enricher.generate_summary("Short")
        
        assert summary is None


class TestEnrichNote:
    """Tests for enrich_note function."""
    
    def create_note(self) -> NoteModel:
        """Helper to create a test note."""
        return NoteModel(
            uuid="test-uuid",
            title="Test Note",
            source_type="web",
            source_path="http://test.com",
            full_text="This is a long article about machine learning and artificial intelligence. " * 20,
            content_hash="abc123",
            metadata=NoteMetadata(),
        )
    
    @patch("zettlecast.enricher.settings")
    def test_enrich_note_disabled(self, mock_settings):
        """Should skip enrichment when disabled."""
        mock_settings.enable_auto_tagging = False
        
        note = self.create_note()
        original_tags = note.metadata.tags.copy()
        
        result = enrich_note(note)
        
        assert result.metadata.tags == original_tags
    
    @patch("zettlecast.enricher.ContentEnricher.extract_tags")
    @patch("zettlecast.enricher.ContentEnricher.generate_summary")
    @patch("zettlecast.enricher.settings")
    def test_enrich_note_adds_tags(self, mock_settings, mock_summary, mock_tags):
        """Should add extracted tags to note metadata."""
        mock_settings.enable_auto_tagging = True
        mock_tags.return_value = ["tag1", "tag2", "tag3"]
        mock_summary.return_value = "A summary sentence."
        
        note = self.create_note()
        result = enrich_note(note)
        
        assert result.metadata.tags == ["tag1", "tag2", "tag3"]
        assert result.metadata.custom["summary"] == "A summary sentence."
    
    @patch("zettlecast.enricher.ContentEnricher.extract_tags")
    @patch("zettlecast.enricher.ContentEnricher.generate_summary")
    @patch("zettlecast.enricher.settings")
    def test_enrich_note_preserves_existing_tags(self, mock_settings, mock_summary, mock_tags):
        """Should not overwrite existing tags."""
        mock_settings.enable_auto_tagging = True
        mock_tags.return_value = ["new-tag"]
        mock_summary.return_value = "Summary"
        
        note = self.create_note()
        note.metadata.tags = ["existing-tag"]
        
        result = enrich_note(note)
        
        # Should not call extract_tags since tags exist
        assert result.metadata.tags == ["existing-tag"]
