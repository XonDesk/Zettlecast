"""
Tests for Aligner Module

Unit tests for RTTM parsing, pyannote annotation handling, and word-speaker alignment.
"""

import pytest

from zettlecast.podcast.aligner import (
    Word,
    SpeakerSegment,
    TranscriptSegment,
    parse_rttm,
    assign_speakers_to_words,
    group_words_by_speaker,
    merge_micro_segments,
    merge_similar_speakers,
    align_transcription_with_diarization,
    parse_pyannote_annotation,
    align_with_pyannote,
)


class TestParseRTTM:
    """Tests for RTTM format parsing."""

    def test_parse_valid_rttm(self):
        """Should parse valid RTTM content."""
        rttm = """SPEAKER audio 1 0.000 5.000 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER audio 1 5.500 4.500 <NA> <NA> SPEAKER_01 <NA> <NA>"""
        
        segments = parse_rttm(rttm)
        
        assert len(segments) == 2
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[0].start == 0.0
        assert segments[0].end == 5.0
        assert segments[1].speaker == "SPEAKER_01"
        assert segments[1].start == 5.5
        assert segments[1].end == 10.0

    def test_parse_empty_rttm(self):
        """Should return empty list for empty content."""
        segments = parse_rttm("")
        assert segments == []

    def test_parse_rttm_ignores_comments(self):
        """Should ignore non-SPEAKER lines."""
        rttm = """# Comment line
SPEAKER audio 1 0.000 5.000 <NA> <NA> SPEAKER_00 <NA> <NA>
some random line"""
        
        segments = parse_rttm(rttm)
        
        assert len(segments) == 1

    def test_parse_rttm_sorts_by_start_time(self):
        """Should sort segments by start time."""
        rttm = """SPEAKER audio 1 5.000 2.000 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER audio 1 0.000 3.000 <NA> <NA> SPEAKER_00 <NA> <NA>"""
        
        segments = parse_rttm(rttm)
        
        assert segments[0].start < segments[1].start


class TestSpeakerSegmentOverlap:
    """Tests for SpeakerSegment overlap detection."""

    def test_overlaps_with_returns_true_for_overlap(self):
        """Should detect overlapping word and speaker segment."""
        segment = SpeakerSegment("SPEAKER_00", 0.0, 5.0)
        word = Word("test", 2.0, 3.0)
        
        assert segment.overlaps_with(word) is True

    def test_overlaps_with_returns_false_for_no_overlap(self):
        """Should return False when no overlap."""
        segment = SpeakerSegment("SPEAKER_00", 0.0, 5.0)
        word = Word("test", 6.0, 7.0)
        
        assert segment.overlaps_with(word) is False

    def test_overlap_duration_calculation(self):
        """Should calculate correct overlap duration."""
        segment = SpeakerSegment("SPEAKER_00", 0.0, 5.0)
        word = Word("test", 3.0, 7.0)  # Partial overlap
        
        duration = segment.overlap_duration(word)
        
        assert duration == 2.0  # Overlap from 3.0 to 5.0


class TestAssignSpeakersToWords:
    """Tests for word-speaker assignment."""

    def test_assigns_speaker_for_overlapping_segment(self):
        """Should assign speaker when word overlaps segment."""
        words = [Word("hello", 1.0, 2.0)]
        segments = [SpeakerSegment("SPEAKER_00", 0.0, 5.0)]
        
        result = assign_speakers_to_words(words, segments)
        
        assert result[0].speaker == "SPEAKER_00"

    def test_assigns_closest_speaker_when_no_overlap(self):
        """Should assign closest speaker when no direct overlap."""
        words = [Word("hello", 10.0, 11.0)]
        segments = [
            SpeakerSegment("SPEAKER_00", 0.0, 5.0),
            SpeakerSegment("SPEAKER_01", 12.0, 15.0),
        ]
        
        result = assign_speakers_to_words(words, segments)
        
        # SPEAKER_01 is closer (starts at 12.0)
        assert result[0].speaker == "SPEAKER_01"

    def test_handles_multiple_overlaps(self):
        """Should assign speaker with maximum overlap when multiple overlap."""
        words = [Word("hello", 4.0, 7.0)]  # Overlaps both segments
        segments = [
            SpeakerSegment("SPEAKER_00", 0.0, 5.0),   # 1s overlap (4-5)
            SpeakerSegment("SPEAKER_01", 5.0, 10.0),  # 2s overlap (5-7)
        ]
        
        result = assign_speakers_to_words(words, segments)
        
        assert result[0].speaker == "SPEAKER_01"


class TestGroupWordsBySpeaker:
    """Tests for grouping words into speaker segments."""

    def test_groups_consecutive_same_speaker(self):
        """Should group consecutive words from same speaker."""
        words = [
            Word("hello", 0.0, 0.5),
            Word("world", 0.6, 1.0),
        ]
        words[0].speaker = "SPEAKER_00"
        words[1].speaker = "SPEAKER_00"
        
        segments = group_words_by_speaker(words)
        
        assert len(segments) == 1
        assert segments[0].text == "hello world"
        assert segments[0].speaker == "SPEAKER_00"

    def test_creates_new_segment_on_speaker_change(self):
        """Should create new segment when speaker changes."""
        words = [
            Word("hello", 0.0, 0.5),
            Word("world", 0.6, 1.0),
        ]
        words[0].speaker = "SPEAKER_00"
        words[1].speaker = "SPEAKER_01"
        
        segments = group_words_by_speaker(words)
        
        assert len(segments) == 2
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[1].speaker == "SPEAKER_01"

    def test_handles_empty_word_list(self):
        """Should return empty list for empty input."""
        segments = group_words_by_speaker([])
        assert segments == []


class TestParsePyannoteAnnotation:
    """Tests for pyannote annotation parsing."""

    def test_parses_annotation_to_speaker_segments(self):
        """Should convert pyannote Annotation to SpeakerSegment list."""
        from unittest.mock import MagicMock
        
        mock_turn1 = MagicMock(start=0.0, end=5.0)
        mock_turn2 = MagicMock(start=5.5, end=10.0)
        
        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = [
            (mock_turn1, None, "SPEAKER_00"),
            (mock_turn2, None, "SPEAKER_01"),
        ]
        
        segments = parse_pyannote_annotation(mock_annotation)
        
        assert len(segments) == 2
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[0].start == 0.0
        assert segments[1].speaker == "SPEAKER_01"

    def test_sorts_by_start_time(self):
        """Should sort segments by start time."""
        from unittest.mock import MagicMock
        
        mock_turn1 = MagicMock(start=5.0, end=10.0)
        mock_turn2 = MagicMock(start=0.0, end=4.0)
        
        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = [
            (mock_turn1, None, "SPEAKER_01"),
            (mock_turn2, None, "SPEAKER_00"),
        ]
        
        segments = parse_pyannote_annotation(mock_annotation)
        
        assert segments[0].start < segments[1].start


class TestAlignWithPyannote:
    """Tests for pyannote-based alignment pipeline."""

    def test_full_alignment_pipeline(self):
        """Should align words with pyannote annotation."""
        from unittest.mock import MagicMock
        
        words = [
            Word("hello", 1.0, 1.5),
            Word("world", 6.0, 6.5),
        ]
        
        mock_turn1 = MagicMock(start=0.0, end=5.0)
        mock_turn2 = MagicMock(start=5.5, end=10.0)
        
        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = [
            (mock_turn1, None, "SPEAKER_00"),
            (mock_turn2, None, "SPEAKER_01"),
        ]
        
        segments = align_with_pyannote(words, mock_annotation)

        assert len(segments) == 2
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[1].speaker == "SPEAKER_01"


class TestMergeSimilarSpeakers:
    """Tests for minor speaker merge post-processing."""

    def _make_word(self, text, start, end, speaker):
        w = Word(text, start, end)
        w.speaker = speaker
        return w

    def _make_segment(self, speaker, words_data):
        """Create a TranscriptSegment from list of (text, start, end) tuples."""
        words = [self._make_word(text, start, end, speaker) for text, start, end in words_data]
        return TranscriptSegment(speaker, words, words[0].start, words[-1].end)

    def test_no_merge_when_all_speakers_significant(self):
        """Should not merge when all speakers have sufficient segments and time."""
        segments = [
            self._make_segment("A", [("hello", 0, 10), ("there", 10, 20)]),
            self._make_segment("B", [("hi", 20, 30), ("back", 30, 40)]),
            self._make_segment("A", [("yes", 40, 50)]),
            self._make_segment("B", [("ok", 50, 60)]),
            self._make_segment("A", [("great", 60, 70)]),
            self._make_segment("B", [("sure", 70, 80)]),
        ]
        result = merge_similar_speakers(segments)
        speakers = set(seg.speaker for seg in result)
        assert speakers == {"A", "B"}

    def test_merges_minor_speaker_by_segment_count(self):
        """Should merge a speaker with fewer than min_speaker_segments."""
        # speaker C has only 1 segment (intro monologue), A and B have many
        segments = [
            self._make_segment("C", [("welcome", 0, 5), ("to", 5, 8), ("the", 8, 10)]),
            self._make_segment("A", [("hello", 15, 25)]),
            self._make_segment("B", [("hi", 25, 35)]),
            self._make_segment("A", [("great", 35, 45)]),
            self._make_segment("B", [("yes", 45, 55)]),
            self._make_segment("A", [("thanks", 55, 65)]),
            self._make_segment("B", [("bye", 65, 75)]),
        ]
        result = merge_similar_speakers(segments)
        speakers = set(seg.speaker for seg in result)
        assert len(speakers) == 2
        assert "C" not in speakers

    def test_merge_target_is_temporally_adjacent(self):
        """Should merge minor speaker into the nearest dominant speaker."""
        # C appears at 0-10s, A starts at 15s, B starts at 100s
        # C should merge into A (temporally closer)
        segments = [
            self._make_segment("C", [("intro", 0, 10)]),
            self._make_segment("A", [("hello", 15, 25)]),
            self._make_segment("B", [("hi", 100, 110)]),
            self._make_segment("A", [("great", 25, 35)]),
            self._make_segment("B", [("yes", 110, 120)]),
            self._make_segment("A", [("thanks", 35, 45)]),
        ]
        result = merge_similar_speakers(segments)
        # C should have been merged into A (the first segment should now be A)
        assert result[0].speaker == "A"
        assert "intro" in result[0].text

    def test_consecutive_segments_consolidated(self):
        """After merge, consecutive same-speaker segments should be combined."""
        # [A] [C] [A] where C is minor -> should become single [A]
        segments = [
            self._make_segment("A", [("first", 0, 10)]),
            self._make_segment("C", [("minor", 10, 12)]),
            self._make_segment("A", [("second", 12, 22)]),
            self._make_segment("B", [("reply", 22, 32)]),
            self._make_segment("A", [("third", 32, 42)]),
            self._make_segment("B", [("end", 42, 52)]),
        ]
        result = merge_similar_speakers(segments)
        # First segment should now contain "first minor second" (all A)
        assert result[0].speaker == "A"
        assert "first" in result[0].text
        assert "minor" in result[0].text
        assert "second" in result[0].text

    def test_preserves_word_speaker_labels(self):
        """Word objects should have their speaker field updated after merge."""
        segments = [
            self._make_segment("C", [("intro", 0, 5)]),
            self._make_segment("A", [("hello", 10, 20)]),
            self._make_segment("B", [("hi", 20, 30)]),
            self._make_segment("A", [("great", 30, 40)]),
            self._make_segment("B", [("ok", 40, 50)]),
            self._make_segment("A", [("bye", 50, 60)]),
        ]
        result = merge_similar_speakers(segments)
        # All words in the merged segment should have updated speaker
        for seg in result:
            for word in seg.words:
                assert word.speaker == seg.speaker

    def test_empty_segments_returns_empty(self):
        """Should handle empty input."""
        result = merge_similar_speakers([])
        assert result == []

    def test_single_speaker_returns_unchanged(self):
        """Should return unchanged when only one speaker."""
        segments = [
            self._make_segment("A", [("hello", 0, 5)]),
            self._make_segment("A", [("world", 5, 10)]),
        ]
        result = merge_similar_speakers(segments)
        assert len(result) == len(segments)
