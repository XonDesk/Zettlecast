"""
Zettlecast Chunking Module
Recursive character-based text splitting.
"""

from typing import List

from .config import settings
from .models import ChunkModel


def recursive_split(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    separators: List[str] = None,
) -> List[str]:
    """
    Recursively split text using multiple separators.
    
    Tries to split on larger semantic boundaries first (paragraphs),
    then falls back to smaller ones (sentences, words).
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    if not text:
        return []
    
    # If text is already small enough, return it
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_separator = separators[0]
    remaining_separators = separators[1:]
    
    # Split on current separator
    parts = text.split(current_separator)
    
    current_chunk = ""
    for part in parts:
        # Add separator back (except for last separator which is empty)
        part_with_sep = part + current_separator if current_separator else part
        
        # If adding this part would exceed chunk size
        if len(current_chunk) + len(part_with_sep) > chunk_size:
            # Save current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # If part itself is too large, recursively split it
            if len(part_with_sep) > chunk_size and remaining_separators:
                sub_chunks = recursive_split(
                    part,
                    chunk_size=chunk_size,
                    chunk_overlap=0,  # No overlap in recursion
                    separators=remaining_separators,
                )
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = part_with_sep
        else:
            current_chunk += part_with_sep
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Apply overlap
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) > chunk_overlap else prev_chunk
                overlapped.append(overlap_text + " " + chunk)
        chunks = overlapped
    
    return chunks


def create_chunks(text: str, base_chunk_id: str) -> List[ChunkModel]:
    """
    Create ChunkModel instances from text.

    Args:
        text: Full text to chunk
        base_chunk_id: Base ID (usually note UUID) for chunk IDs

    Returns:
        List of ChunkModel instances
    """
    raw_chunks = recursive_split(text)

    chunks = []
    current_pos = 0

    for i, chunk_text in enumerate(raw_chunks):
        # Find position in original text
        start_char = text.find(chunk_text[:50], current_pos)  # Use first 50 chars to find
        if start_char == -1:
            start_char = current_pos
        end_char = start_char + len(chunk_text)
        current_pos = end_char

        # Skip very small chunks, but keep at least one chunk if we only have one
        if len(chunk_text.strip()) < settings.min_chunk_size and len(raw_chunks) > 1:
            continue

        chunk = ChunkModel(
            chunk_id=f"{base_chunk_id}_chunk_{i}",
            text=chunk_text,
            start_char=start_char,
            end_char=end_char,
        )
        chunks.append(chunk)

    return chunks


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count (4 chars per token on average).
    """
    return len(text) // 4
