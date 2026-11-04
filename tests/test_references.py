"""References: links and names found in a note, Wikipedia lookup, fetching a page as a linked note."""

import httpx

from zettlecast import references
from zettlecast.linker import wikilink_pairs
from zettlecast.models import NoteMetadata, NoteModel, ProcessingResult
from zettlecast.podcast.rss import html_to_text


def note(text="", custom=None, source="https://www.reddit.com/r/running/comments/p1/x", title="Heat training"):
    return NoteModel(uuid="src-1", title=title, source_type="reddit", source_path=source, full_text=text,
                     content_hash="h", metadata=NoteMetadata(source_url=source, custom=custom or {}))


def test_links_markdown_reddit_style_and_bare():
    text = ("See [the study](https://example.com/study?utm_source=x) first.\n"
            "A guide (https://blog.example.org/guide/) helped.\n"
            "Also https://other.example.net/page.\n"
            "Again https://example.com/study")
    links = references.extract_links(note(text))
    assert [link["url"] for link in links] == [
        "https://example.com/study?utm_source=x", "https://blog.example.org/guide/", "https://other.example.net/page"]
    assert links[0]["label"] == "the study"
    assert links[1]["label"] == "A guide"
    assert links[2]["label"] == "other.example.net/page"
    assert "first" in links[0]["context"]


def test_links_drop_invisible_characters():
    links = references.extract_links(note("Survey: https://forms.gle/abc⁠⁠ thanks"))
    assert links[0]["url"] == "https://forms.gle/abc"


def test_links_skip_media_reddit_own_source_and_rank_low_value_last():
    text = ("https://youtu.be/abc and https://i.imgur.com/x.jpg and https://www.reddit.com/r/a/comments/b\n"
            "https://www.reddit.com/r/running/comments/p1/x/ and https://example.com/article")
    links = references.extract_links(note(text))
    assert [(link["domain"], link["low_value"]) for link in links] == [("example.com", False), ("youtu.be", True)]


def test_links_ignore_screen_text_but_keep_thread_comments():
    text = ("Focus\n\n## Screen Text\n\nhttps://ads.example.com/x\n\n"
            "## Thread comments\n\nworth it (https://example.com/c)\n")
    assert [link["url"] for link in references.extract_links(note(text))] == ["https://example.com/c"]


def test_reddit_link_post_url():
    links = references.extract_links(note("text", custom={"reddit": {"link_url": "https://example.com/a/"}}))
    assert links[0]["url"] == "https://example.com/a/"


def test_names_from_kept_item_people_and_organisations():
    custom = {"saved_as": "media", "item": {"name": "Dune", "by": "Frank Herbert", "type": "book"},
              "people": ["Frank Herbert", "Jason Koop"], "organisations": ["UESCA"]}
    names = references.extract_names(note(custom=custom))
    assert [(n["label"], n["query"], n["what"]) for n in names] == [
        ("Dune", "Dune Frank Herbert", "book"), ("Frank Herbert", "Frank Herbert", "person"),
        ("Jason Koop", "Jason Koop", "person"), ("UESCA", "UESCA", "organisation")]


def test_names_skip_items_not_worth_looking_up():
    names = references.extract_names(note(custom={"saved_as": "quote", "item": {"name": "Some quote"}}))
    assert names == []


def test_lookup_parses_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(references.settings, "storage_path", tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"query": {"pages": {
            "2": {"title": "Dune (film)", "index": 2, "description": "2021 film of the Frank Herbert novel",
                  "fullurl": "https://en.wikipedia.org/wiki/Dune_(film)"},
            "3": {"title": "Frank Ocean", "index": 3, "description": "American singer",
                  "fullurl": "https://en.wikipedia.org/wiki/Frank_Ocean"},
            "1": {"title": "Dune (novel)", "index": 1, "description": "1965 novel by Frank Herbert",
                  "fullurl": "https://en.wikipedia.org/wiki/Dune_(novel)"}}}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = references.lookup("Dune Frank Herbert", client=client)
    assert [c["title"] for c in first] == ["Dune (novel)", "Dune (film)"]
    assert first[0]["description"] == "1965 novel by Frank Herbert"
    assert references.lookup("dune frank herbert ", client=client) == first
    assert len(calls) == 1


def test_lookup_drops_pages_that_share_one_word(tmp_path, monkeypatch):
    monkeypatch.setattr(references.settings, "storage_path", tmp_path)
    pages = {"1": {"title": "Judy Baar Topinka", "index": 1, "description": "American politician",
                   "fullurl": "https://en.wikipedia.org/wiki/Judy_Baar_Topinka"}}
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"query": {"pages": pages}})))
    assert references.lookup("Keith Baar", client=client) == []


def test_fetch_links_back_to_source(monkeypatch):
    stored, requested = [], []
    fetched = NoteModel(uuid="new-1", title="The study", source_type="web",
                        source_path="https://example.com/study/", full_text="Body", content_hash="x")
    monkeypatch.setattr(references.db, "web_source_paths", lambda: [])

    def parse(url):
        requested.append(url)
        return ProcessingResult(status="success", uuid="new-1", title="The study", note=fetched)

    monkeypatch.setattr("zettlecast.parser.parse_url", parse)
    monkeypatch.setattr("zettlecast.pipeline.ingest_note", lambda n: stored.append(n) or n)

    result = references.fetch(note(title="Untitled"), "https://example.com/study/?sig=a%2Fb")
    assert result.status == "success" and result.uuid == "new-1"
    assert requested == ["https://example.com/study/?sig=a%2Fb"]  # the link as given, not its normalised key
    assert stored[0].full_text.endswith("Referenced from Untitled\n")  # no [[wikilink]] matched by title
    assert stored[0].metadata.custom["referenced_by"] == ["src-1"]


def test_fetch_existing_note_is_a_duplicate_linked_to_the_new_source(monkeypatch):
    existing = NoteModel(uuid="old-1", title="Old", source_type="web",
                         source_path="https://www.example.com/a/?utm_source=bookmarklet",
                         full_text="x", content_hash="x", metadata=NoteMetadata(custom={"referenced_by": ["other"]}))
    saved, relinked = [], []
    monkeypatch.setattr(references.db, "web_source_paths", lambda: [("old-1", existing.source_path)])
    monkeypatch.setattr(references.db, "get_note_by_uuid", lambda uuid: existing if uuid == "old-1" else None)
    monkeypatch.setattr(references.db, "update_note_metadata", lambda uuid, meta: saved.append((uuid, meta)))
    monkeypatch.setattr("zettlecast.linker.update_edges_for_note", lambda uuid: relinked.append(uuid))

    result = references.fetch(note(), "https://www.example.com/a")
    assert (result.status, result.uuid) == ("duplicate", "old-1")
    assert saved[0][1]["custom"]["referenced_by"] == ["other", "src-1"]
    assert relinked == ["old-1"]

    references.fetch(note(), "https://www.example.com/a")  # already referenced from it: nothing to do
    assert len(saved) == 1


def test_extract_marks_links_already_stored_under_a_raw_url(monkeypatch):
    monkeypatch.setattr(references.db, "web_source_paths",
                        lambda: [("old-1", "https://example.com/post/?utm_source=x")])
    links = [r for r in references.extract(note("See https://example.com/post")) if r["kind"] == "url"]
    assert links[0]["existing_uuid"] == "old-1"


def test_links_keep_parentheses_in_urls():
    text = ("[Dune](https://en.wikipedia.org/wiki/Dune_(novel)) and the film "
            "(https://en.wikipedia.org/wiki/Dune_(2021_film)), also https://en.wikipedia.org/wiki/Arrakis_(planet).")
    assert [link["url"] for link in references.extract_links(note(text))] == [
        "https://en.wikipedia.org/wiki/Dune_(novel)", "https://en.wikipedia.org/wiki/Dune_(2021_film)",
        "https://en.wikipedia.org/wiki/Arrakis_(planet)"]


def test_referenced_by_is_a_link_edge():
    notes = [{"uuid": "a", "title": "Comment by u/x: [Q] #1", "full_text": "", "metadata": {}},
             {"uuid": "b", "title": "Page", "full_text": "Referenced from Comment by u/x: [Q] #1",
              "metadata": {"custom": {"referenced_by": ["a"]}}}]
    assert wikilink_pairs(notes) == {(0, 1)}


def test_show_notes_keep_links():
    html = '<p>Read <a href="https://example.com/book">the book</a> and https://plain.example.com</p>'
    assert html_to_text(html) == "Read the book (https://example.com/book) and https://plain.example.com"


def test_show_notes_skip_a_url_that_is_its_own_label():
    assert html_to_text('<p><a href="https://www.example.com/">example.com</a></p>') == "example.com"


def test_link_urls_are_not_indexed():
    from zettlecast.chunker import strip_link_urls
    assert strip_link_urls("Read the book (https://example.com/book) now") == "Read the book now"
    assert strip_link_urls("Bare https://example.com stays") == "Bare https://example.com stays"


def test_rejected_pair_overrides_a_referenced_by_link_but_not_a_wikilink():
    import numpy as np
    from zettlecast.linker import Graph, select_pairs

    notes = [{"uuid": "a"}, {"uuid": "b"}, {"uuid": "c"}]
    # a-b from referenced_by, a-c a [[wikilink]] typed by the user
    graph = Graph(notes, np.zeros((3, 3)), None, None, None, None, None, {(0, 1), (0, 2)}, {(0, 2)})
    assert select_pairs(graph, set()) == {(0, 1), (0, 2)}
    assert select_pairs(graph, {("b", "a"), ("a", "c")}) == {(0, 2)}
    assert select_pairs(graph, {("a", "b"), ("c", "a")}, only=0, kth={}) == {(0, 2)}


def test_wikilink_pairs_without_referenced_by():
    notes = [{"uuid": "a", "title": "Heat", "full_text": "", "metadata": {}},
             {"uuid": "b", "title": "Page", "full_text": "see [[Heat]]", "metadata": {}},
             {"uuid": "c", "title": "Other", "full_text": "", "metadata": {"custom": {"referenced_by": ["a"]}}}]
    assert wikilink_pairs(notes) == {(0, 1), (0, 2)}
    assert wikilink_pairs(notes, referenced=False) == {(0, 1)}
