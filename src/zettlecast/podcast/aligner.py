"""
Word-to-Speaker Alignment Module

Merges word-level transcription timestamps with speaker diarization labels.
This is the "glue" script between Parakeet transcription and MSDD diarization.
"""

import logging
from typing import List, Tuple

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

    logger.info("Pyannote alignment complete")
    return transcript_segments

