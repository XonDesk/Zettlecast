"""Queue run: every episode is transcribed before any is enhanced; saved transcriptions are reused."""

from pathlib import Path
from unittest.mock import patch

from zettlecast import pipeline
from zettlecast.models import NoteModel
from zettlecast.podcast.asr_cache import asr_path, load_asr, save_asr
from zettlecast.podcast.models import TranscriptionResult, TranscriptSegment
from zettlecast.podcast.queue import TranscriptionQueue


def result(episode_id="x"):
    seg = TranscriptSegment(start=0, end=1, text="hello there", speaker="speaker_0")
    return TranscriptionResult(episode_id=episode_id, segments=[seg], full_text="[0.0s - 1.0s] speaker_0: hello there",
                               duration_seconds=1.0, processing_time_seconds=0.1)


class FakeTranscriber:
    def __init__(self, events):
        self.events = events

    def transcribe(self, path, episode=None, progress_callback=None):
        self.events.append(("transcribe", Path(path).name))
        return result(episode.id)

    def cleanup_gpu(self):
        self.events.append(("cleanup", ""))


def run(tmp_path, queue, events):
    def save_result(res, episode, enhanced):
        events.append(("enhance", Path(episode.audio_path).name))
        note = tmp_path / f"{episode.id}.md"
        note.write_text("x", encoding="utf-8")
        return note

    parsed = type("P", (), {"status": "success", "note": NoteModel(title="t", full_text="x", source_type="audio",
                                                                    source_path="x", content_hash="h")})()
    with patch("zettlecast.podcast.queue.TranscriptionQueue", return_value=queue), \
         patch("zettlecast.podcast.nemo_transcriber.NeMoTranscriber",
               side_effect=lambda: FakeTranscriber(events)), \
         patch("zettlecast.podcast.formatter.save_result", side_effect=save_result), \
         patch.object(pipeline, "parse_file", return_value=parsed), \
         patch.object(pipeline.db, "upsert_note"), \
         patch.object(pipeline, "update_edges_for_note"), \
         patch("zettlecast.llm.unload"):
        return pipeline._run_podcast_queue(enhance=False, sync=False)


def make_queue(tmp_path, n=2):
    queue = TranscriptionQueue(state_file=tmp_path / "queue.json")
    for i in range(n):
        audio = tmp_path / f"ep{i}.mp3"
        audio.write_bytes(bytes([i]) * 100)
        queue.add(audio, podcast_name="Show", episode_title=f"Episode {i}")
    return queue


def test_all_episodes_are_transcribed_before_any_is_enhanced(tmp_path):
    events = []
    summary = run(tmp_path, make_queue(tmp_path), events)
    assert [e[0] for e in events] == ["transcribe", "transcribe", "cleanup", "enhance", "enhance"]
    assert summary.completed == 2
    assert asr_path(tmp_path / "ep0.mp3").exists()


def test_saved_transcription_is_reused(tmp_path):
    queue = make_queue(tmp_path)
    save_asr(tmp_path / "ep0.mp3", result())
    events = []
    run(tmp_path, queue, events)
    assert ("transcribe", "ep0.mp3") not in events and ("transcribe", "ep1.mp3") in events


def test_saved_transcription_from_other_models_is_ignored(tmp_path, monkeypatch):
    save_asr(tmp_path / "ep0.mp3", result())
    assert load_asr(tmp_path / "ep0.mp3") is not None
    monkeypatch.setattr(pipeline.settings, "nemo_parakeet_model", "nvidia/some-other-model")
    assert load_asr(tmp_path / "ep0.mp3") is None


def test_has_asr_reads_only_the_header(tmp_path, monkeypatch):
    from zettlecast.podcast.asr_cache import has_asr

    assert not has_asr(tmp_path / "ep0.mp3")
    save_asr(tmp_path / "ep0.mp3", result())
    assert has_asr(tmp_path / "ep0.mp3")
    monkeypatch.setattr(pipeline.settings, "nemo_parakeet_model", "nvidia/some-other-model")
    assert not has_asr(tmp_path / "ep0.mp3")
