"""Tests for RSS show notes flowing into the queue, enhancer and transcript note."""

from unittest.mock import patch

import pytest

from zettlecast.podcast.enhancer import TranscriptEnhancer
from zettlecast.podcast.formatter import format_transcript_for_zettlecast
from zettlecast.podcast.models import TranscriptionResult
from zettlecast.podcast.queue import DuplicateEpisodeError, TranscriptionQueue
from zettlecast.podcast.rss import html_to_text

NOTES = "Baptiste Morel-Prieur is an author on a recent paper."


def test_html_to_text_keeps_paragraphs_and_list_items():
    html = (
        "<p>Baptiste  Morel-Prieur is a <b>researcher</b>.<br>See <a href='x'>the paper</a></p>"
        "<ul><li>UTMB data</li><li>Gradients</li></ul>"
    )
    assert html_to_text(html) == (
        "Baptiste Morel-Prieur is a researcher.\nSee the paper\nUTMB data\nGradients"
    )


@pytest.fixture
def queue(tmp_path):
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"fake audio")
    return TranscriptionQueue(state_file=tmp_path / "queue.json"), audio


def test_import_adds_show_notes_to_completed_episode(queue):
    q, audio = queue
    job_id = q.add(audio, podcast_name="KoopCast")
    q.items[job_id].status = "completed"
    q.completed_hashes.append(q.items[job_id].episode.audio_hash)

    with pytest.raises(DuplicateEpisodeError):
        q.add(audio, feed_url="https://feed", description=NOTES, published="Fri, 16 May 2025")

    reloaded = TranscriptionQueue(state_file=q.state_file)
    episode = reloaded.items[job_id].episode
    assert episode.description == NOTES
    assert episode.feed_url == "https://feed"
    assert episode.published == "Fri, 16 May 2025"


def test_reprocess_requeues_matching_completed_episodes(queue):
    q, audio = queue
    job_id = q.add(audio, episode_title="Pacing with Baptiste")
    q.items[job_id].status = "completed"

    assert q.reprocess("nothing matches") == 0
    assert q.reprocess("baptiste") == 1
    assert q.items[job_id].status == "pending"


def test_analysis_prompt_includes_show_notes():
    with patch("zettlecast.podcast.enhancer.chat", return_value={"speakers": {}}) as chat:
        TranscriptEnhancer().analyze("[0.0s - 1.0s] speaker_0: hi", show_notes=NOTES)
    prompt = chat.call_args.args[0]
    assert "Show notes (from the podcast feed):" in prompt
    assert NOTES in prompt


def test_analysis_prompt_without_show_notes():
    with patch("zettlecast.podcast.enhancer.chat", return_value={"speakers": {}}) as chat:
        TranscriptEnhancer().analyze("[0.0s - 1.0s] speaker_0: hi")
    assert "Show notes" not in chat.call_args.args[0]
    assert "earlier episodes of this show:" not in chat.call_args.args[0]


def test_analysis_prompt_includes_known_speakers():
    with patch("zettlecast.podcast.enhancer.chat", return_value={"speakers": {}}) as chat:
        TranscriptEnhancer().analyze("[0.0s - 1.0s] speaker_0: hi", known_speakers=["Adam Ferdinandson"])
    assert "People on earlier episodes of this show: Adam Ferdinandson" in chat.call_args.args[0]


def test_known_speakers_from_other_episodes_of_the_show(tmp_path, monkeypatch):
    from zettlecast import pipeline

    def note(name, uuid, show, speakers):
        names = "\n".join(f"  speaker_{i}: {n}" for i, n in enumerate(speakers))
        (tmp_path / name).write_text(
            f"---\nuuid: {uuid}\npodcast:\n  show: {show}\nspeaker_names:\n{names}\n---\nbody\n",
            encoding="utf-8",
        )

    note("a.md", "a", "KoopCast", ["Jason Koop", "Adam Ferdinandson"])
    note("b.md", "b", "KoopCast", ["Jason Koop", "Adam"])
    note("c.md", "c", "Other Show", ["Someone Else"])
    monkeypatch.setattr(type(pipeline.settings), "transcripts_path", property(lambda self: tmp_path))

    assert pipeline.known_speakers("KoopCast") == ["Jason Koop", "Adam Ferdinandson"]
    assert pipeline.known_speakers("KoopCast", exclude_uuid="a") == ["Jason Koop"]
    assert pipeline.known_speakers("") == []


def test_note_contains_show_notes_section(queue):
    q, audio = queue
    job_id = q.add(audio, episode_title="Ep", description=NOTES, published="Fri, 16 May 2025")
    result = TranscriptionResult(
        episode_id=job_id, full_text="[0.0s - 1.0s] speaker_0: hi",
        duration_seconds=1, processing_time_seconds=1,
    )
    md = format_transcript_for_zettlecast(result, q.items[job_id].episode)
    assert f"## Show Notes\n\n{NOTES}" in md
    assert "published: Fri, 16 May 2025" in md
    assert md.index("## Show Notes") < md.index("## Transcript")


def test_known_speakers_prefers_verified_names(tmp_path, monkeypatch):
    from zettlecast import pipeline

    def note(name, speaker, verified=False):
        lines = ["---", f"uuid: {name}", "podcast:", "  show: KoopCast",
                 "speaker_names:", f"  speaker_0: {speaker}"]
        if verified:
            lines.append(f"verified_speakers: [{speaker}]")
        (tmp_path / f"{name}.md").write_text("\n".join(lines + ["---", ""]), encoding="utf-8")

    note("a", "Ryan Anderson")
    note("b", "Ryne Anderson", verified=True)
    note("c", "Ryan Anderson")
    monkeypatch.setattr(type(pipeline.settings), "transcripts_path", property(lambda self: tmp_path))
    assert pipeline.known_speakers("KoopCast") == ["Ryne Anderson"]
