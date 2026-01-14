"""
Zettlecast Tests - Identity Module
"""

import tempfile
from pathlib import Path

import pytest

from zettlecast.identity import (
    compute_content_hash,
    ensure_uuid_in_content,
    ensure_uuid_in_file,
    generate_uuid,
    get_uuid_from_content,
    parse_frontmatter,
    serialize_frontmatter,
)


class TestUUIDGeneration:
    def test_generate_uuid_format(self):
        uuid = generate_uuid()
        assert len(uuid) == 36
        assert uuid.count("-") == 4

    def test_generate_uuid_unique(self):
        uuids = [generate_uuid() for _ in range(100)]
        assert len(set(uuids)) == 100


class TestContentHash:
    def test_hash_consistency(self):
        content = "Hello, world!"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2

    def test_hash_different_content(self):
        hash1 = compute_content_hash("Hello")
        hash2 = compute_content_hash("World")
        assert hash1 != hash2


class TestFrontmatterParsing:
    def test_parse_basic_frontmatter(self):
        content = """---
title: Test Note
uuid: 12345
---

Body content here."""
        
        frontmatter, body = parse_frontmatter(content)
        
        assert frontmatter["title"] == "Test Note"
        assert frontmatter["uuid"] == "12345"
        assert "Body content" in body

    def test_parse_no_frontmatter(self):
        content = "Just some plain text."
        
        frontmatter, body = parse_frontmatter(content)
        
        assert frontmatter == {}
        assert body == content

    def test_serialize_frontmatter(self):
        data = {"title": "Test", "uuid": "abc123"}
        yaml_str = serialize_frontmatter(data)
        
        assert "title: Test" in yaml_str
        assert "uuid: abc123" in yaml_str


class TestEnsureUUID:
    def test_adds_uuid_to_content_without_frontmatter(self):
        content = "Just plain text."
        
        new_content, uuid, modified = ensure_uuid_in_content(content)
        
        assert modified is True
        assert uuid is not None
        assert len(uuid) == 36
        assert "---" in new_content
        assert uuid in new_content

    def test_preserves_existing_uuid(self):
        content = """---
uuid: existing-uuid-here
---

Body text."""
        
        new_content, uuid, modified = ensure_uuid_in_content(content)
        
        assert modified is False
        assert uuid == "existing-uuid-here"
        assert new_content == content

    def test_adds_uuid_to_existing_frontmatter(self):
        content = """---
title: My Note
---

Body text."""
        
        new_content, uuid, modified = ensure_uuid_in_content(content)
        
        assert modified is True
        assert uuid is not None
        assert "uuid:" in new_content
        assert "title: My Note" in new_content


class TestFileOperations:
    def test_ensure_uuid_in_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test Note\n\nSome content.")
            temp_path = Path(f.name)
        
        try:
            uuid, modified = ensure_uuid_in_file(temp_path)
            
            assert modified is True
            assert uuid is not None
            
            # Read back and verify
            content = temp_path.read_text()
            assert uuid in content
            assert "---" in content
        finally:
            temp_path.unlink()

    def test_get_uuid_from_content(self):
        content = """---
uuid: test-uuid-123
---

Content."""
        
        uuid = get_uuid_from_content(content)
        assert uuid == "test-uuid-123"

    def test_get_uuid_from_content_none(self):
        content = "No frontmatter here."
        uuid = get_uuid_from_content(content)
        assert uuid is None
