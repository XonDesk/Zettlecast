"""Tests for LLM tag/summary enrichment (LLM calls mocked)."""

from unittest.mock import patch

import pytest

from zettlecast.enricher import enrich_note
from zettlecast.llm import LLMError
from zettlecast.models import NoteMetadata, NoteModel

TEXT = "Gut training improves carbohydrate absorption during ultramarathons. " * 5


def make_note(**metadata) -> NoteModel:
    return NoteModel(
        title="Gut training",
        source_type="pdf",
        source_path="x.pdf",
        full_text=TEXT,
        content_hash="h",
        metadata=NoteMetadata(**metadata),
    )


@pytest.fixture(autouse=True)
def tagging_on():
    with patch("zettlecast.enricher.settings") as s:
        s.enable_auto_tagging = True
        s.auto_tag_count = 3
        yield s


def test_adds_normalised_tags_and_summary():
    reply = {"tags": [" Gut Training ", "Nutrition", "", "ultra", "extra"], "summary": " Gut training helps. "}
    with patch("zettlecast.enricher.chat", return_value=reply):
        note = enrich_note(make_note())
    assert note.metadata.tags == ["gut training", "nutrition", "ultra"]
    assert note.metadata.custom["summary"] == "Gut training helps."


def test_preserves_existing_tags():
    with patch("zettlecast.enricher.chat", return_value={"tags": ["new"], "summary": "S"}):
        note = enrich_note(make_note(tags=["mine"]))
    assert note.metadata.tags == ["mine"]
    assert note.metadata.custom["summary"] == "S"


def test_skips_when_already_enriched():
    with patch("zettlecast.enricher.chat") as chat:
        enrich_note(make_note(tags=["a"], custom={"summary": "done"}))
    chat.assert_not_called()


def test_disabled(tagging_on):
    tagging_on.enable_auto_tagging = False
    with patch("zettlecast.enricher.chat") as chat:
        note = enrich_note(make_note())
    chat.assert_not_called()
    assert note.metadata.tags == []


def test_llm_failure_leaves_note_unchanged():
    with patch("zettlecast.enricher.chat", side_effect=LLMError("down")):
        note = enrich_note(make_note())
    assert note.metadata.tags == []
    assert "summary" not in note.metadata.custom


def test_keeps_full_names_and_organisations_for_graph_links():
    reply = {"tags": ["gut training"], "summary": "S", "people": ["Isabel Martinez", "Fred", " "],
             "organisations": ["Monash University", "Monash University"]}
    with patch("zettlecast.enricher.chat", return_value=reply):
        note = enrich_note(make_note())
    assert note.metadata.custom["people"] == ["Isabel Martinez"]
    assert note.metadata.custom["organisations"] == ["Monash University"]
