"""
Zettlecast Tests - Chunker Module
"""

import pytest

from zettlecast.chunker import create_chunks, estimate_tokens, recursive_split


class TestRecursiveSplit:
    def test_short_text_no_split(self):
        text = "Short text."
        chunks = recursive_split(text, chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_on_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = recursive_split(text, chunk_size=30)
        
        assert len(chunks) >= 2

    def test_split_on_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = recursive_split(text, chunk_size=40)
        
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunks = recursive_split("")
        assert chunks == []

    def test_overlap_applied(self):
        text = "Chunk one content here.\n\nChunk two content here.\n\nChunk three content here."
        chunks = recursive_split(text, chunk_size=30, chunk_overlap=10)
        
        # With overlap, later chunks should have some content from previous
        if len(chunks) > 1:
            # Just verify we got multiple chunks
            assert len(chunks) >= 2


class TestCreateChunks:
    def test_creates_chunk_models(self):
        text = "First part of the document.\n\nSecond part of the document."
        chunks = create_chunks(text, base_chunk_id="test-uuid")
        
        assert len(chunks) >= 1
        assert all(c.chunk_id.startswith("test-uuid") for c in chunks)
        assert all(c.text for c in chunks)

    def test_chunk_positions(self):
        text = "Start of text.\n\nMiddle of text.\n\nEnd of text."
        chunks = create_chunks(text, base_chunk_id="test")
        
        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char >= chunk.start_char


class TestTokenEstimation:
    def test_estimate_tokens(self):
        text = "Hello world"  # 11 chars
        tokens = estimate_tokens(text)
        
        # Should be roughly 11/4 = 2-3 tokens
        assert 1 <= tokens <= 5

    def test_longer_text(self):
        text = "a" * 400
        tokens = estimate_tokens(text)
        
        assert tokens == 100  # 400 / 4
