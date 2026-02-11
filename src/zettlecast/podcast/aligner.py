"""
Word-to-Speaker Alignment Module

Merges word-level transcription timestamps with speaker diarization labels.
This is the "glue" script between Parakeet transcription and MSDD diarization.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Word:
    """Word with timestamp from transcription."""

    def __init__(self, text: str, start: float, end: float):
        self.text = text
        self.start = start
        self.end = end
        self.speaker = None  # Assigned during alignment

    def __repr__(self):
        return f"Word('{self.text}', {self.start:.2f}-{self.end:.2f}, {self.speaker})"


class SpeakerSegment:
    """Speaker segment from diarization (RTTM format)."""

    def __init__(self, speaker: str, start: float, end: float):
        self.speaker = speaker
        self.start = start
        self.end = end

    def overlaps_with(self, word: Word) -> bool:
        """Check if this speaker segment overlaps with a word's timestamp."""
        # Word overlaps if any part of it falls within speaker segment
        return not (word.end <= self.start or word.start >= self.end)

    def overlap_duration(self, word: Word) -> float:
        """Calculate overlap duration between speaker segment and word."""
        overlap_start = max(self.start, word.start)
        overlap_end = min(self.end, word.end)
        return max(0.0, overlap_end - overlap_start)

    def __repr__(self):
        return f"SpeakerSegment({self.speaker}, {self.start:.2f}-{self.end:.2f})"


class TranscriptSegment:
    """A segment of consecutive words from the same speaker."""

    def __init__(self, speaker: str, words: List[Word], start: float, end: float):
        self.speaker = speaker
        self.words = words
        self.start = start
        self.end = end
        self.text = " ".join(w.text for w in words)

    def __repr__(self):
        return f"TranscriptSegment({self.speaker}, {self.start:.2f}-{self.end:.2f})"


def parse_rttm(rttm_content: str) -> List[SpeakerSegment]:
    """
    Parse RTTM format speaker diarization output.

    RTTM Format:
    SPEAKER <file> 1 <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>

    Args:
        rttm_content: String content of RTTM file

    Returns:
        List of SpeakerSegment objects sorted by start time
    """
    segments = []

    for line in rttm_content.strip().split("\n"):
        if not line.strip() or not line.startswith("SPEAKER"):
            continue

        parts = line.split()
        if len(parts) < 8:
            logger.warning(f"Invalid RTTM line: {line}")
            continue

        start_time = float(parts[3])
        duration = float(parts[4])
        speaker_id = parts[7]

        end_time = start_time + duration
        segments.append(SpeakerSegment(speaker_id, start_time, end_time))

    segments.sort(key=lambda s: s.start)
    logger.info(f"Parsed {len(segments)} speaker segments from RTTM")
    return segments


def assign_speakers_to_words(
    words: List[Word], speaker_segments: List[SpeakerSegment]
) -> List[Word]:
    """
    Assign speaker labels to words based on timestamp overlap.

    Args:
        words: List of words with timestamps
        speaker_segments: List of speaker segments from diarization

    Returns:
        Same list of words with speaker field populated
    """
    if not speaker_segments:
        logger.warning("No speaker segments provided, words will have no speaker labels")
        return words

    for word in words:
        # Find all overlapping speaker segments
        overlapping = [seg for seg in speaker_segments if seg.overlaps_with(word)]

        if not overlapping:
            # No overlap - assign to closest speaker segment
            closest = min(
                speaker_segments,
                key=lambda s: min(abs(s.start - word.start), abs(s.end - word.end)),
            )
            word.speaker = closest.speaker
            logger.debug(f"Word '{word.text}' has no overlap, assigned to closest: {closest.speaker}")
        elif len(overlapping) == 1:
            # Single overlap - straightforward assignment
            word.speaker = overlapping[0].speaker
        else:
            # Multiple overlaps - assign to segment with maximum overlap
            best_segment = max(overlapping, key=lambda s: s.overlap_duration(word))
            word.speaker = best_segment.speaker
            logger.debug(
                f"Word '{word.text}' overlaps multiple speakers, "
                f"assigned to {best_segment.speaker}"
            )

    return words


def group_words_by_speaker(words: List[Word]) -> List[TranscriptSegment]:
    """
    Group consecutive words from the same speaker into segments.

    Args:
        words: List of words with speaker labels assigned

    Returns:
        List of TranscriptSegment objects
    """
    if not words:
        return []

    segments = []
    current_speaker = words[0].speaker
    current_words = [words[0]]
    segment_start = words[0].start

    for word in words[1:]:
        if word.speaker == current_speaker:
            # Continue current segment
            current_words.append(word)
        else:
            # Speaker changed - finalize current segment
            segment_end = current_words[-1].end
            segments.append(
                TranscriptSegment(current_speaker, current_words, segment_start, segment_end)
            )

            # Start new segment
            current_speaker = word.speaker
            current_words = [word]
            segment_start = word.start

    # Finalize last segment
    segment_end = current_words[-1].end
    segments.append(
        TranscriptSegment(current_speaker, current_words, segment_start, segment_end)
    )

    logger.info(f"Grouped {len(words)} words into {len(segments)} speaker segments")
    return segments


def merge_micro_segments(
    segments: List[TranscriptSegment],
    min_duration: float = 1.5,
) -> List[TranscriptSegment]:
    """
    Merge very short segments into their neighbors to reduce diarization noise.

    Short segments (< min_duration) that are sandwiched between segments of the
    same speaker are almost always diarization artifacts. Merge them into the
    surrounding speaker's segment.

    Args:
        segments: List of transcript segments
        min_duration: Segments shorter than this (seconds) are candidates for merging

    Returns:
        List of merged transcript segments
    """
    if len(segments) <= 2:
        return segments

    merged = list(segments)
    changed = True

    while changed:
        changed = False
        new_merged = []
        i = 0
        while i < len(merged):
            seg = merged[i]
            duration = seg.end - seg.start

            # Check if this is a micro-segment sandwiched between same-speaker segments
            if (
                duration < min_duration
                and i > 0
                and i < len(merged) - 1
                and new_merged
                and new_merged[-1].speaker == merged[i + 1].speaker
            ):
                # Merge into the previous segment (absorb the micro-segment's words)
                prev = new_merged[-1]
                combined_words = prev.words + seg.words
                new_merged[-1] = TranscriptSegment(
                    prev.speaker, combined_words, prev.start, seg.end
                )
                changed = True
                logger.debug(
                    f"Merged micro-segment ({duration:.1f}s, '{seg.text[:30]}...') "
                    f"into {prev.speaker}"
                )
            else:
                new_merged.append(seg)
            i += 1

        # Also merge consecutive segments with the same speaker
        final = []
        for seg in new_merged:
            if final and final[-1].speaker == seg.speaker:
                prev = final[-1]
                combined_words = prev.words + seg.words
                final[-1] = TranscriptSegment(
                    prev.speaker, combined_words, prev.start, seg.end
                )
                changed = True
            else:
                final.append(seg)

        merged = final

    if len(merged) < len(segments):
        logger.info(
            f"Merged {len(segments) - len(merged)} micro-segments "
            f"({len(segments)} -> {len(merged)} segments)"
        )
    return merged


def merge_similar_speakers(
    segments: List[TranscriptSegment],
    min_speaker_segments: int = 3,
    min_speaker_time_ratio: float = 0.05,
) -> List[TranscriptSegment]:
    """
    Merge minor speakers into dominant speakers to fix over-segmentation.

    When diarization splits one real speaker into two labels (e.g., host
    during solo intro vs. host during conversation), the minor label
    typically has very few segments or appears only in a narrow time window.

    A speaker is "minor" if ANY of these are true:
      - Has fewer than min_speaker_segments segments
      - Has less than min_speaker_time_ratio of total speaking time

    The minor speaker is merged into the temporally nearest dominant speaker.

    Args:
        segments: List of TranscriptSegment objects
        min_speaker_segments: Speakers with fewer segments are merge candidates
        min_speaker_time_ratio: Speakers with less than this fraction of
            total speaking time are merge candidates

    Returns:
        List of TranscriptSegment with minor speakers relabeled and
        consecutive same-speaker segments merged.
    """
    if not segments:
        return segments

    # Compute per-speaker stats
    speaker_stats: Dict[str, dict] = {}
    for seg in segments:
        duration = seg.end - seg.start
        if seg.speaker not in speaker_stats:
            speaker_stats[seg.speaker] = {
                "segment_count": 0,
                "total_duration": 0.0,
                "earliest_start": seg.start,
                "latest_end": seg.end,
            }
        stats = speaker_stats[seg.speaker]
        stats["segment_count"] += 1
        stats["total_duration"] += duration
        stats["earliest_start"] = min(stats["earliest_start"], seg.start)
        stats["latest_end"] = max(stats["latest_end"], seg.end)

    # Need at least 2 speakers to merge
    if len(speaker_stats) < 2:
        return segments

    total_duration = sum(s["total_duration"] for s in speaker_stats.values())
    if total_duration == 0:
        return segments

    # Identify minor speakers
    minor_speakers = set()
    for speaker, stats in speaker_stats.items():
        if (
            stats["segment_count"] < min_speaker_segments
            or stats["total_duration"] / total_duration < min_speaker_time_ratio
        ):
            minor_speakers.add(speaker)

    # Don't merge if all speakers are minor or none are
    dominant_speakers = set(speaker_stats.keys()) - minor_speakers
    if not minor_speakers or not dominant_speakers:
        return segments

    # For each minor speaker, find best merge target
    merge_map: Dict[str, str] = {}
    for minor in minor_speakers:
        minor_stats = speaker_stats[minor]
        minor_start = minor_stats["earliest_start"]
        minor_end = minor_stats["latest_end"]

        # Find temporally adjacent dominant speakers (within 30s)
        candidates = []
        for dom in dominant_speakers:
            dom_stats = speaker_stats[dom]
            # Check if any dominant segments are near the minor speaker's time range
            gap = max(
                0,
                max(minor_start - dom_stats["latest_end"], dom_stats["earliest_start"] - minor_end),
            )
            candidates.append((dom, gap, dom_stats["total_duration"]))

        # Sort by proximity first, then by speaking time (prefer more dominant)
        candidates.sort(key=lambda c: (c[1], -c[2]))
        merge_map[minor] = candidates[0][0]

        logger.info(
            f"Merging minor speaker {minor} -> {candidates[0][0]} "
            f"({minor_stats['segment_count']} segments, "
            f"{minor_stats['total_duration']:.1f}s)"
        )

    # Relabel segments and their words
    for seg in segments:
        if seg.speaker in merge_map:
            new_speaker = merge_map[seg.speaker]
            seg.speaker = new_speaker
            for word in seg.words:
                word.speaker = new_speaker

    # Consolidate consecutive same-speaker segments
    consolidated = []
    for seg in segments:
        if consolidated and consolidated[-1].speaker == seg.speaker:
            prev = consolidated[-1]
            combined_words = prev.words + seg.words
            consolidated[-1] = TranscriptSegment(
                prev.speaker, combined_words, prev.start, seg.end
            )
        else:
            consolidated.append(seg)

    if len(consolidated) < len(segments):
        logger.info(
            f"Speaker merge: {len(segments)} -> {len(consolidated)} segments "
            f"({len(speaker_stats)} -> {len(dominant_speakers)} speakers)"
        )

    return consolidated


def align_transcription_with_diarization(
    words: List[Word], rttm_content: str
) -> List[TranscriptSegment]:
    """
    Complete alignment pipeline: parse RTTM, assign speakers, and group words.

    Args:
        words: List of words with timestamps from transcription
        rttm_content: RTTM format string from diarization

    Returns:
        List of TranscriptSegment objects with speaker labels
    """
    logger.info(f"Starting alignment for {len(words)} words")

    # Step 1: Parse RTTM
    speaker_segments = parse_rttm(rttm_content)

    # Step 2: Assign speakers to words
    words_with_speakers = assign_speakers_to_words(words, speaker_segments)

    # Step 3: Group consecutive words by speaker
    transcript_segments = group_words_by_speaker(words_with_speakers)

    # Step 4: Merge micro-segments to reduce diarization noise
    transcript_segments = merge_micro_segments(transcript_segments)

    # Step 5: Merge minor speakers (reduces over-segmentation)
    transcript_segments = merge_similar_speakers(transcript_segments)

    logger.info("Alignment complete")
    return transcript_segments


def parse_pyannote_annotation(annotation) -> List[SpeakerSegment]:
    """
    Convert pyannote.core.Annotation to SpeakerSegment list.

    This is an adapter for Mac transcriber which uses pyannote.audio
    for diarization instead of NeMo MSDD.

    Args:
        annotation: pyannote.core.Annotation object

    Returns:
        List of SpeakerSegment objects sorted by start time
    """
    segments = []

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(SpeakerSegment(
            speaker=speaker,
            start=turn.start,
            end=turn.end,
        ))

    segments.sort(key=lambda s: s.start)
    logger.info(f"Parsed {len(segments)} speaker segments from pyannote annotation")
    return segments


def align_with_pyannote(
    words: List[Word], annotation
) -> List[TranscriptSegment]:
    """
    Alignment pipeline using pyannote annotation instead of RTTM.

    Args:
        words: List of words with timestamps from transcription
        annotation: pyannote.core.Annotation object from diarization

    Returns:
        List of TranscriptSegment objects with speaker labels
    """
    logger.info(f"Starting pyannote alignment for {len(words)} words")

    # Step 1: Parse pyannote annotation
    speaker_segments = parse_pyannote_annotation(annotation)

    # Step 2: Assign speakers to words
    words_with_speakers = assign_speakers_to_words(words, speaker_segments)

    # Step 3: Group consecutive words by speaker
    transcript_segments = group_words_by_speaker(words_with_speakers)

    # Step 4: Merge micro-segments to reduce diarization noise
    transcript_segments = merge_micro_segments(transcript_segments)

    # Step 5: Merge minor speakers (reduces over-segmentation)
    transcript_segments = merge_similar_speakers(transcript_segments)

    logger.info("Pyannote alignment complete")
    return transcript_segments

