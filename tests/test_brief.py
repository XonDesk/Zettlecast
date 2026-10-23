"""The short description the graph shows for a note."""

from zettlecast.brief import BRIEF_CHARS, note_brief


def test_the_llm_summary_comes_first():
    assert note_brief({"summary": "The summary."}, "## Post\nThe post.") == "The summary."


def test_a_screenshot_has_its_description():
    assert note_brief({"description": "A comment recommending Dune."}, "## Kept\nBook: Dune") \
        == "A comment recommending Dune."


def test_without_the_text_only_summary_or_description_count():
    assert note_brief({"summary": ""}) == ""


def test_untagged_reddit_saves_show_the_saved_comment_or_the_post():
    post = "r/running · posted by u/a · 2026-01-01\n\n## Post\nWhich shoes\nfor trails?\n\n## Thread comments\nLots."
    assert note_brief({"summary": ""}, post) == "Which shoes for trails?"
    saved = "r/running\n\n## Saved comment\nThe one I kept.\n\n## The post\nThe question."
    assert note_brief({}, saved) == "The one I kept."


def test_otherwise_the_start_without_headings_shortened_at_a_word():
    brief = note_brief({}, "# Title\n\n" + "word " * 500)
    assert brief.startswith("word word") and brief.endswith("…")
    assert len(brief) <= BRIEF_CHARS + 1


def test_only_screenshots_kept_with_their_image_show_it():
    from zettlecast.brief import has_image

    assert has_image("image", {"image_kept": True})
    assert not has_image("image", {"image_kept": False})  # kept for its information only
    assert not has_image("audio", {})
