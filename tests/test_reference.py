"""Tests for published-transcript parsing, speaker naming and comparison."""

import json
from unittest.mock import patch

from zettlecast.podcast.enhancer import TranscriptEnhancer
from zettlecast.podcast.reference import (
    compare,
    is_named,
    note_turns,
    parse_reference,
    speaker_names_from_reference,
)
from zettlecast.podcast.rss import transcript_urls


def ref_json(segments):
    return json.dumps({"version": "1.0.0", "segments": [
        {"speaker": sp, "startTime": s, "endTime": e, "body": b} for sp, s, e, b in segments
    ]})


def test_parse_json_splits_bodies_into_timed_words():
    words = parse_reference(ref_json([("Jason Koop", 0.0, 2.0, "hello there"), ("Jason Koop", 2.0, 2.1, " ")]))
    assert [(w.text, w.start, w.end, w.speaker) for w in words] == [
        ("hello", 0.0, 1.0, "Jason Koop"), ("there", 1.0, 2.0, "Jason Koop"),
    ]


def test_parse_vtt_with_voice_tags():
    vtt = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:03.000\n<v Ryne Anderson>Hi all</v>\n\n00:04.000 --> 00:05.000\nplain\n"
    words = parse_reference(vtt)
    assert [(w.text, w.speaker) for w in words] == [("Hi", "Ryne Anderson"), ("all", "Ryne Anderson"), ("plain", None)]
    assert words[0].start == 1.0 and words[-1].start == 4.0


def test_parse_srt():
    srt = "1\n00:00:01,500 --> 00:00:02,500\nhello world\n"
    assert [(w.text, w.start) for w in parse_reference(srt)] == [("hello", 1.5), ("world", 2.0)]


def test_generic_speaker_labels_are_not_names():
    assert not is_named("Speaker 1")
    assert not is_named("SPEAKER_00")
    assert not is_named(None)
    assert is_named("Benat Mencia")


TURNS = [
    (0.0, 10.0, "speaker_0", "welcome to the koopcast i'm jason"),
    (10.0, 20.0, "speaker_1", "thanks for having me on the show"),
]


def words(speaker, start, end, n):
    return [(speaker, start + i * (end - start) / n, start + (i + 1) * (end - start) / n, f"w{i}") for i in range(n)]


def test_speaker_names_from_named_reference():
    ref = parse_reference(ref_json(words("Jason Koop", 0, 10, 30) + words("Benat Mencia", 10, 20, 30)))
    assert speaker_names_from_reference(TURNS, ref) == {"speaker_0": "Jason Koop", "speaker_1": "Benat Mencia"}


def test_generic_reference_speakers_give_no_names():
    ref = parse_reference(ref_json(words("Speaker 1", 0, 10, 30) + words("Speaker 2", 10, 20, 30)))
    assert speaker_names_from_reference(TURNS, ref) == {}


def test_too_little_overlap_gives_no_name():
    ref = parse_reference(ref_json(words("Jason Koop", 0, 10, 5)))
    assert speaker_names_from_reference(TURNS, ref) == {}


NOTE = """---
title: x
---

## Summary

text

## Transcript

[0.0s - 10.0s] Jason Koop: Welcome to the Coop Cast, I'm Jason.
[10.0s - 20.0s] Ryan Anderson: Thanks for, um, having me on the show.
"""


def test_note_turns():
    assert note_turns(NOTE) == [
        (0.0, 10.0, "Jason Koop", "Welcome to the Coop Cast, I'm Jason."),
        (10.0, 20.0, "Ryan Anderson", "Thanks for, um, having me on the show."),
    ]


def test_compare_reports_differences_and_names():
    ref = parse_reference(ref_json([
        ("Jason Koop", 0.0, 10.0, "Welcome to the KoopCast, I'm Jason."),
        ("Ryne Anderson", 10.0, 20.0, "Thanks for having me on the show."),
    ]))
    r = compare(note_turns(NOTE), ref)
    assert r.reference_words == 13
    assert r.substitutions == 1 and r.extra == 1 and r.missing == 0  # koopcast -> coop cast
    assert r.speaker_agreement == 1.0
    assert r.name_mismatches == [("Ryne Anderson", "Ryan Anderson")]
    assert r.top_differences[0][0] == "koopcast"
    assert r.spans and "koopcast" in r.spans[0].reference and r.spans[0].time == 5.0


def test_transcript_urls_prefers_json():
    xml = b"""<rss xmlns:podcast="https://podcastindex.org/namespace/1.0"><channel>
      <item><guid>a</guid>
        <podcast:transcript url="https://x/a.vtt" type="text/vtt"/>
        <podcast:transcript url="https://x/a.json" type="application/json"/></item>
      <item><guid>b</guid><podcast:transcript url="https://x/b.srt" type="application/x-subrip"/></item>
      <item><guid>c</guid></item>
    </channel></rss>"""
    assert transcript_urls(xml) == {"a": "https://x/a.json", "b": "https://x/b.srt"}


def test_published_speakers_override_llm_and_feed_cleanup_vocabulary():
    analysis = {"keywords": [], "summary": "", "key_points": [], "chapters": [],
                "speakers": {"speaker_0": "Jason Koop", "speaker_1": "Ryan Anderson"}}
    transcript = "[0.0s - 1.0s] speaker_0: hi\n[1.0s - 2.0s] speaker_1: hello"
    with patch("zettlecast.podcast.enhancer.chat", side_effect=[analysis, transcript]) as chat:
        out = TranscriptEnhancer().enhance(
            transcript, podcast="KoopCast", show_notes="Notes here",
            published_speakers={"speaker_1": "Ryne Anderson"},
        )
    assert out["speakers"] == {"speaker_0": "Jason Koop", "speaker_1": "Ryne Anderson"}
    assert "speaker_1 = Ryne Anderson" in chat.call_args_list[0].args[0]
    cleanup_prompt = chat.call_args_list[1].args[0]
    assert "Correct spellings for this episode" in cleanup_prompt
    assert "Podcast: KoopCast" in cleanup_prompt and "Show notes: Notes here" in cleanup_prompt
    assert "[1.0s - 2.0s] Ryne Anderson: hello" in out["cleaned_transcript"]


# --- Name guard and snapping ---

from zettlecast.podcast.enhancer import guard_names, snap_name  # noqa: E402


def test_guard_reverts_swapped_name():
    orig = "[1.0s - 2.0s] speaker_1: the Norwegian triathletes Christian Blumfeld and Gustav Ili and others"
    cleaned = "[1.0s - 2.0s] speaker_1: the Norwegian triathletes Christian Blumfeld and Gustav Ingebrigtsen and others"
    out, reverted = guard_names(orig, cleaned, allowed=set())
    assert out == orig
    assert reverted == [{"text": "Ili (cleanup suggested: Ingebrigtsen)"}]


def test_guard_keeps_vocabulary_names_joins_and_normal_fixes():
    orig = "[1.0s - 2.0s] speaker_0: welcome to the Coop Cast with Ryan and um lactaid threshold. so"
    cleaned = "[1.0s - 2.0s] speaker_0: welcome to the KoopCast with Ryne and lactate threshold. So"
    out, reverted = guard_names(orig, cleaned, allowed={"koopcast", "ryne", "anderson"})
    assert out == cleaned
    assert reverted == []


def test_guard_keeps_case_only_and_joined_words():
    orig = "[1.0s - 2.0s] speaker_0: logged in training peaks and strava"
    cleaned = "[1.0s - 2.0s] speaker_0: logged in TrainingPeaks and Strava"
    assert guard_names(orig, cleaned, allowed=set()) == (cleaned, [])


def test_snap_name():
    known = ["Ryne Anderson", "Adam Ferdinandson"]
    assert snap_name("Ryan Anderson", known) == "Ryne Anderson"
    assert snap_name("Adam", known) == "Adam"
    assert snap_name("Sophie Herzog", known) == "Sophie Herzog"


def test_enhance_snaps_llm_name_and_fixes_summary():
    analysis = {
        "keywords": ["ryan anderson"], "summary": "Coach Ryan Anderson explains.", "key_points": [],
        "chapters": [{"name": "Ryan Anderson on data", "start_time": 0, "description": "x"}],
        "speakers": {"speaker_0": "Jason Koop", "speaker_3": "Ryan Anderson"},
    }
    with patch("zettlecast.podcast.enhancer.chat", return_value=analysis):
        out = TranscriptEnhancer().enhance("[0.0s - 1.0s] speaker_3: hi", cleanup=False,
                                           known_speakers=["Ryne Anderson"])
    assert out["speakers"]["speaker_3"] == "Ryne Anderson"
    assert out["summary"] == "Coach Ryne Anderson explains."
    assert out["keywords"] == ["ryne anderson"]
    assert out["sections"][0]["name"] == "Ryne Anderson on data"
    assert out["verified_speakers"] == []
