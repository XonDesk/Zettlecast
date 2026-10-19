"""Tests for podcast transcript enhancement (LLM calls mocked)."""

from unittest.mock import patch

from zettlecast.llm import LLMError
from zettlecast.podcast.enhancer import CLEANUP_PROMPT, TranscriptEnhancer

TRANSCRIPT = (
    "[0.0s - 5.0s] speaker_0: Welcome to the show, I'm your host Jason.\n"
    "[5.2s - 9.0s] speaker_1: Thanks for having me, um, Jason.\n"
    "[9.1s - 12.0s] speaker_0: Let's talk about lactaid threshold."
)

ANALYSIS = {
    "keywords": ["Lactate Threshold", " endurance "],
    "summary": "Jason talks about lactate threshold.",
    "key_points": ["Point one"],
    "chapters": [{"name": "Intro", "start_time": 0, "description": "Welcome"}],
    "speakers": {"speaker_0": "Jason Koop", "speaker_1": None},
}


class TestCleanupPrompt:
    def test_mentions_domain_corrections(self):
        assert "domain-specific" in CLEANUP_PROMPT
        assert "speaker" in CLEANUP_PROMPT.lower()

    def test_has_uncertainty_instructions(self):
        assert "[[corrected text??]]" in CLEANUP_PROMPT


class TestExtractUncertainCorrections:
    def test_no_markers_returns_unchanged(self):
        cleaned, items = TranscriptEnhancer.extract_uncertain_corrections("plain text")
        assert cleaned == "plain text"
        assert items == []

    def test_markers_removed_and_positions_tracked(self):
        text = "He ran the [[Western States??]] and [[Hardrock??]]."
        cleaned, items = TranscriptEnhancer.extract_uncertain_corrections(text)
        assert cleaned == "He ran the Western States and Hardrock."
        assert [i["text"] for i in items] == ["Western States", "Hardrock"]
        for item in items:
            assert cleaned[item["position"]:].startswith(item["text"])


class TestCleanupValidation:
    def test_accepts_same_lines_and_prefixes(self):
        cleaned = TRANSCRIPT.replace("um, ", "").replace("lactaid", "lactate")
        assert TranscriptEnhancer._is_valid_cleanup_response(TRANSCRIPT, cleaned)

    def test_rejects_lost_line(self):
        cleaned = "\n".join(TRANSCRIPT.split("\n")[:2])
        assert not TranscriptEnhancer._is_valid_cleanup_response(TRANSCRIPT, cleaned)

    def test_rejects_changed_prefix(self):
        cleaned = TRANSCRIPT.replace("[5.2s - 9.0s] speaker_1:", "[5.2s - 9.0s] Guest:")
        assert not TranscriptEnhancer._is_valid_cleanup_response(TRANSCRIPT, cleaned)

    def test_rejects_placeholder(self):
        assert not TranscriptEnhancer._is_valid_cleanup_response(TRANSCRIPT, "Please provide the transcript.")


class TestCleanupTranscript:
    def test_keeps_original_when_structure_breaks(self):
        with patch("zettlecast.podcast.enhancer.chat", return_value="Sure! Here it is."):
            cleaned, items = TranscriptEnhancer().cleanup_transcript(TRANSCRIPT)
        assert cleaned == TRANSCRIPT
        assert items == []

    def test_keeps_original_on_llm_error(self):
        with patch("zettlecast.podcast.enhancer.chat", side_effect=LLMError("down")):
            cleaned, _ = TranscriptEnhancer().cleanup_transcript(TRANSCRIPT)
        assert cleaned == TRANSCRIPT

    def test_chunks_long_transcripts(self):
        long = "\n".join([TRANSCRIPT] * 20)
        with patch("zettlecast.podcast.enhancer.chat", side_effect=lambda p: p.split("Transcript chunk:\n", 1)[1]) as chat:
            cleaned, _ = TranscriptEnhancer().cleanup_transcript(long, chunk_size=500)
        assert chat.call_count > 1
        assert cleaned == long


class TestParallelChunks:
    def test_chat_each_keeps_order_and_returns_errors(self):
        import time

        from zettlecast.llm import chat_each

        def call(i):
            def run():
                time.sleep(0.05 * (3 - i))  # later calls finish first
                if i == 1:
                    raise LLMError("down")
                return i
            return run

        with patch("zettlecast.llm.settings.llm_parallel", 4):
            out = list(chat_each(call(i) for i in range(4)))
        assert out[0] == 0 and isinstance(out[1], LLMError) and out[2:] == [2, 3]

    def test_cleanup_chunks_come_back_in_order(self):
        import random
        import time

        def fake_chat(prompt, **_):
            time.sleep(random.random() * 0.05)
            return prompt.split("Transcript chunk:\n", 1)[1].replace("lactaid", "lactate")

        long = "\n".join([TRANSCRIPT] * 20)
        with patch("zettlecast.podcast.enhancer.chat", side_effect=fake_chat), \
                patch("zettlecast.llm.settings.llm_parallel", 4):
            cleaned, _ = TranscriptEnhancer().cleanup_transcript(long, chunk_size=500)
        assert cleaned == long.replace("lactaid", "lactate")


class TestApplySpeakerNames:
    def test_replaces_only_identified_labels(self):
        out = TranscriptEnhancer.apply_speaker_names(TRANSCRIPT, {"speaker_0": "Jason Koop"})
        assert "] Jason Koop: Welcome" in out
        assert "] speaker_1: Thanks" in out
        assert "speaker_0" not in out


class TestEnhance:
    def test_returns_all_fields_with_names_applied(self):
        def fake_chat(prompt, schema=None, **_):
            if schema:
                return dict(ANALYSIS)
            return prompt.split("Transcript chunk:\n", 1)[1].replace("lactaid", "lactate")

        with patch("zettlecast.podcast.enhancer.chat", side_effect=fake_chat):
            result = TranscriptEnhancer().enhance(TRANSCRIPT, podcast="KoopCast", episode="Ep 1")

        assert result["keywords"] == ["lactate threshold", "endurance"]
        assert result["speakers"] == {"speaker_0": "Jason Koop"}
        assert result["summary"] == ANALYSIS["summary"]
        assert result["sections"] == ANALYSIS["chapters"]
        assert "lactate threshold" in result["cleaned_transcript"]
        assert "] Jason Koop: Welcome" in result["cleaned_transcript"]

    def test_people_and_organisations_use_known_spellings(self):
        analysis = {**ANALYSIS, "people": ["Jason Koop", "Jasn Koop", "Fred", "  "], "organisations": ["UTMB", "UTMB"]}
        with patch("zettlecast.podcast.enhancer.chat", return_value=analysis):
            result = TranscriptEnhancer().enhance(TRANSCRIPT, cleanup=False)
        assert result["people"] == ["Jason Koop"]
        assert result["organisations"] == ["UTMB"]

    def test_without_cleanup_keeps_text(self):
        with patch("zettlecast.podcast.enhancer.chat", return_value=dict(ANALYSIS)):
            result = TranscriptEnhancer().enhance(TRANSCRIPT, cleanup=False)
        assert "lactaid" in result["cleaned_transcript"]

    def test_analysis_failure_degrades_gracefully(self):
        with patch("zettlecast.podcast.enhancer.chat", side_effect=LLMError("down")):
            result = TranscriptEnhancer().enhance(TRANSCRIPT)
        assert result["keywords"] == []
        assert result["cleaned_transcript"] == TRANSCRIPT
