"""
Unit tests for the linker module.
Tests composite weighting algorithm, Jaccard similarity, and temporal direction.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from zettlecast.linker import (
    calculate_jaccard_similarity,
    calculate_composite_weight,
    get_temporal_direction,
    GraphEdge,
)
from zettlecast.models import NoteModel, NoteMetadata


class TestJaccardSimilarity:
    """Tests for calculate_jaccard_similarity function."""
    
    def test_identical_tags(self):
        """Identical tag sets should return 1.0."""
        tags = ["python", "machine-learning", "ai"]
        assert calculate_jaccard_similarity(tags, tags) == 1.0
    
    def test_no_overlap(self):
        """Completely different tags should return 0.0."""
        tags_a = ["python", "javascript"]
        tags_b = ["rust", "golang"]
        assert calculate_jaccard_similarity(tags_a, tags_b) == 0.0
    
    def test_partial_overlap(self):
        """Partial overlap should return correct ratio."""
        tags_a = ["python", "javascript", "react"]
        tags_b = ["python", "react", "vue"]
        # Intersection: python, react = 2
        # Union: python, javascript, react, vue = 4
        # Jaccard = 2/4 = 0.5
        assert calculate_jaccard_similarity(tags_a, tags_b) == 0.5
    
    def test_empty_tags(self):
        """Empty tag sets should return 0.0."""
        assert calculate_jaccard_similarity([], []) == 0.0
        assert calculate_jaccard_similarity(["python"], []) == 0.0
        assert calculate_jaccard_similarity([], ["python"]) == 0.0
    
    def test_case_insensitive(self):
        """Tags should be compared case-insensitively."""
        tags_a = ["Python", "REACT"]
        tags_b = ["python", "react"]
        assert calculate_jaccard_similarity(tags_a, tags_b) == 1.0
    
    def test_whitespace_handling(self):
        """Whitespace should be stripped from tags."""
        tags_a = ["  python  ", "react"]
        tags_b = ["python", "  react  "]
        assert calculate_jaccard_similarity(tags_a, tags_b) == 1.0


class TestCompositeWeight:
    """Tests for calculate_composite_weight function."""
    
    def test_pure_vector_sim(self):
        """With beta=0, only vector similarity matters."""
        result = calculate_composite_weight(0.8, 0.5, alpha=1.0, beta=0.0)
        assert result == 0.8
    
    def test_pure_tag_sim(self):
        """With alpha=0, only tag similarity matters."""
        result = calculate_composite_weight(0.8, 0.5, alpha=0.0, beta=1.0)
        assert result == 0.5
    
    def test_default_weights(self):
        """Default 0.7/0.3 split should work correctly."""
        # 0.7 * 0.8 + 0.3 * 0.5 = 0.56 + 0.15 = 0.71
        result = calculate_composite_weight(0.8, 0.5, alpha=0.7, beta=0.3)
        assert result == 0.71
    
    def test_negative_vector_sim_clamped(self):
        """Negative vector similarity should be clamped to 0."""
        result = calculate_composite_weight(-0.5, 0.5, alpha=0.5, beta=0.5)
        # -0.5 clamped to 0, so 0.5 * 0 + 0.5 * 0.5 = 0.25
        assert result == 0.25
    
    def test_weights_normalized(self):
        """Weights should be normalized if they don't sum to 1."""
        # alpha=2, beta=2 should normalize to 0.5/0.5
        result = calculate_composite_weight(0.8, 0.4, alpha=2.0, beta=2.0)
        # 0.5 * 0.8 + 0.5 * 0.4 = 0.4 + 0.2 = 0.6
        assert result == 0.6


class TestTemporalDirection:
    """Tests for get_temporal_direction function."""
    
    def create_note(self, uuid: str, created_at: datetime) -> NoteModel:
        """Helper to create a mock note."""
        return NoteModel(
            uuid=uuid,
            title="Test",
            source_type="web",
            source_path="http://test.com",
            full_text="Test content",
            content_hash="abc123",
            created_at=created_at,
            metadata=NoteMetadata(),
        )
    
    @patch("zettlecast.linker.settings")
    def test_older_to_newer(self, mock_settings):
        """Older note should point to newer note."""
        mock_settings.graph_temporal_direction = True
        
        old_note = self.create_note("old-uuid", datetime(2024, 1, 1))
        new_note = self.create_note("new-uuid", datetime(2024, 6, 1))
        
        source, target, is_directed = get_temporal_direction(old_note, new_note)
        
        assert source == "old-uuid"
        assert target == "new-uuid"
        assert is_directed is True
    
    @patch("zettlecast.linker.settings")
    def test_newer_to_older_reversed(self, mock_settings):
        """Arguments in reverse order should still produce older→newer."""
        mock_settings.graph_temporal_direction = True
        
        old_note = self.create_note("old-uuid", datetime(2024, 1, 1))
        new_note = self.create_note("new-uuid", datetime(2024, 6, 1))
        
        source, target, is_directed = get_temporal_direction(new_note, old_note)
        
        assert source == "old-uuid"
        assert target == "new-uuid"
        assert is_directed is True
    
    @patch("zettlecast.linker.settings")
    def test_undirected_mode(self, mock_settings):
        """With temporal direction disabled, edges are undirected."""
        mock_settings.graph_temporal_direction = False
        
        note_a = self.create_note("b-uuid", datetime(2024, 1, 1))
        note_b = self.create_note("a-uuid", datetime(2024, 6, 1))
        
        source, target, is_directed = get_temporal_direction(note_a, note_b)
        
        # Should be alphabetically ordered
        assert source == "a-uuid"
        assert target == "b-uuid"
        assert is_directed is False


class TestGraphEdge:
    """Tests for GraphEdge class."""
    
    def test_to_dict(self):
        """to_dict should return all fields."""
        edge = GraphEdge(
            source_uuid="source",
            target_uuid="target",
            weight=0.85,
            vector_sim=0.9,
            tag_sim=0.7,
            is_directed=True,
            edge_type="semantic",
        )
        
        d = edge.to_dict()
        
        assert d["source_uuid"] == "source"
        assert d["target_uuid"] == "target"
        assert d["weight"] == 0.85
        assert d["vector_sim"] == 0.9
        assert d["tag_sim"] == 0.7
        assert d["is_directed"] is True
        assert d["edge_type"] == "semantic"
        assert "created_at" in d
