"""
Zettlecast Identity Manager
Handles UUID assignment and frontmatter management.
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

import yaml

from .config import settings


# Regex to match YAML frontmatter
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def generate_uuid() -> str:
    """Generate a new UUID v4."""
    return str(uuid4())


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def parse_frontmatter(content: str) -> Tuple[dict, str]:
    """
    Parse YAML frontmatter from markdown content.

    Returns:
        Tuple of (frontmatter dict, body content)
    """
    match = FRONTMATTER_PATTERN.match(content)
    if match:
        try:
            frontmatter = yaml.safe_load(match.group(1)) or {}
            body = content[match.end():]

            # Normalize uuid field to string if present
            # (YAML auto-types numeric values like "12345" as int)
            if "uuid" in frontmatter and not isinstance(frontmatter["uuid"], str):
                frontmatter["uuid"] = str(frontmatter["uuid"])

            return frontmatter, body
        except yaml.YAMLError:
            # Malformed YAML, return empty frontmatter
            return {}, content
    return {}, content


def serialize_frontmatter(frontmatter: dict) -> str:
    """Serialize frontmatter dict to YAML string."""
    return yaml.dump(frontmatter, default_flow_style=False, sort_keys=False, allow_unicode=True)


def ensure_uuid_in_content(content: str, title: Optional[str] = None) -> Tuple[str, str, bool]:
    """
    Ensure content has a UUID in frontmatter.
    
    Args:
        content: Markdown content (may or may not have frontmatter)
        title: Optional title to add if creating new frontmatter
        
    Returns:
        Tuple of (updated_content, uuid, was_modified)
    """
    frontmatter, body = parse_frontmatter(content)
    
    # Check if UUID already exists
    existing_uuid = frontmatter.get("uuid")
    if existing_uuid:
        return content, existing_uuid, False
    
    # Generate new UUID and add to frontmatter
    new_uuid = generate_uuid()
    frontmatter["uuid"] = new_uuid
    
    # Add title if provided and not present
    if title and "title" not in frontmatter:
        frontmatter["title"] = title
    
    # Add timestamp if not present
    if "created" not in frontmatter:
        frontmatter["created"] = datetime.utcnow().isoformat()
    
    # Ensure status is set
    if "status" not in frontmatter:
        frontmatter["status"] = "inbox"
    
    # Reconstruct content with updated frontmatter
    new_content = f"---\n{serialize_frontmatter(frontmatter)}---\n{body}"
    
    return new_content, new_uuid, True


def ensure_uuid_in_file(file_path: Path) -> Tuple[str, bool]:
    """
    Ensure a markdown file has a UUID in its frontmatter.
    Writes back to file if modified.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Tuple of (uuid, was_modified)
    """
    content = file_path.read_text(encoding="utf-8")
    
    # Extract title from filename if needed
    title = file_path.stem.replace("_", " ").replace("-", " ")
    
    new_content, uuid, was_modified = ensure_uuid_in_content(content, title)
    
    if was_modified:
        # Atomic write: write to temp file, then rename
        temp_path = file_path.with_suffix(".tmp")
        temp_path.write_text(new_content, encoding="utf-8")
        temp_path.rename(file_path)
    
    return uuid, was_modified


def get_uuid_from_content(content: str) -> Optional[str]:
    """Extract UUID from content frontmatter if present."""
    frontmatter, _ = parse_frontmatter(content)
    return frontmatter.get("uuid")


def get_uuid_from_file(file_path: Path) -> Optional[str]:
    """Extract UUID from a file's frontmatter if present."""
    try:
        content = file_path.read_text(encoding="utf-8")
        return get_uuid_from_content(content)
    except Exception:
        return None


def create_note_file(
    uuid: str,
    title: str,
    content: str,
    source_type: str,
    source_path: str,
    metadata: Optional[dict] = None,
) -> Path:
    """
    Create a new note file with proper frontmatter.
    
    Args:
        uuid: Note UUID
        title: Note title
        content: Note body content
        source_type: Type of source (pdf, web, audio, etc.)
        source_path: Original source path or URL
        metadata: Additional metadata to include
        
    Returns:
        Path to the created file
    """
    settings.ensure_directories()
    
    # Build frontmatter
    frontmatter = {
        "uuid": uuid,
        "title": title,
        "source_type": source_type,
        "source": source_path,
        "status": "inbox",
        "created": datetime.utcnow().isoformat(),
    }
    
    if metadata:
        frontmatter.update(metadata)
    
    # Create file content
    file_content = f"---\n{serialize_frontmatter(frontmatter)}---\n\n{content}"
    
    # Sanitize title for filename
    safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip().replace(" ", "_")
    filename = f"{safe_title}_{uuid[:8]}.md"
    
    file_path = settings.storage_path / filename
    file_path.write_text(file_content, encoding="utf-8")
    
    return file_path


def add_wikilink_to_file(file_path: Path, target_title: str) -> bool:
    """
    Add a wikilink to a note file.
    
    Args:
        file_path: Path to the note file
        target_title: Title of the note to link to
        
    Returns:
        True if link was added, False if already exists
    """
    content = file_path.read_text(encoding="utf-8")
    
    # Check if link already exists
    wikilink = f"[[{target_title}]]"
    if wikilink in content:
        return False
    
    # Add link at the end of the file
    if not content.endswith("\n"):
        content += "\n"
    
    content += f"\n## Related\n\n- {wikilink}\n"
    
    file_path.write_text(content, encoding="utf-8")
    return True
