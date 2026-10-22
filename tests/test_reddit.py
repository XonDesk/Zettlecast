"""Reddit saves: Atom parsing, feed paging, note text, one note per save, incremental sync, pacing."""

import time

import httpx
import pytest

from zettlecast import reddit

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
  <author><name>/u/op</name><uri>https://www.reddit.com/user/op</uri></author>
  <category term="running" label="r/running"/>
  <content type="html">&lt;!-- SC_OFF --&gt;&lt;div class="md"&gt;&lt;p&gt;Body with a &lt;a href="https://example.com/a"&gt;source&lt;/a&gt;.&lt;/p&gt;&lt;ul&gt;&lt;li&gt;point&lt;/li&gt;&lt;/ul&gt;&lt;/div&gt;&lt;!-- SC_ON --&gt; &amp;#32; submitted by &amp;#32; &lt;a href="https://www.reddit.com/user/op"&gt; /u/op &lt;/a&gt; &lt;br/&gt; &lt;span&gt;&lt;a href="https://example.com/study"&gt;[link]&lt;/a&gt;&lt;/span&gt; &amp;#32; &lt;span&gt;&lt;a href="https://www.reddit.com/r/running/comments/p1/x/"&gt;[comments]&lt;/a&gt;&lt;/span&gt;</content>
  <id>t3_p1</id>
  <link href="https://www.reddit.com/r/running/comments/p1/x/"/>
  <published>2024-05-01T10:00:00+00:00</published>
  <title>Heat training study</title>
</entry>
<entry>
  <author><name>/u/alice</name></author>
  <category term="running" label="r/running"/>
  <content type="html">&lt;!-- SC_OFF --&gt;&lt;div class="md"&gt;&lt;p&gt;A comment&lt;/p&gt;&lt;/div&gt;&lt;!-- SC_ON --&gt;</content>
  <id>t1_c1</id>
  <link href="https://www.reddit.com/r/running/comments/p1/x/c1/"/>
  <updated>2024-05-02T10:00:00+00:00</updated>
  <title>/u/alice on Heat training study</title>
</entry>
</feed>"""


def item(name, text="", author="op", title=None, link=None, sub="running"):
    kind_post = name.startswith("t3_")
    permalink = f"https://www.reddit.com/r/{sub}/comments/p1/x/" + ("" if kind_post else f"{name[3:]}/")
    return {"name": name, "title": title or (f"Post {name}" if kind_post else f"/u/{author} on Post t3_p1"),
            "author": author, "subreddit": sub, "permalink": permalink, "date": "2024-05-01T10:00:00",
            "text": text, "link": link}


class FakeClient:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get_feed(self, url, params=None):
        self.calls.append((url, params))
        for key, value in self.routes.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value(params) if callable(value) else value
        raise reddit.RedditError(f"HTTP 404 for {url}")


THREAD = [item("t3_p1", "Body of the post."), item("t1_a", "Top comment", author="alice"),
          item("t1_m", "Rules reminder", author="AutoModerator"), item("t1_d", "[deleted]", author="x"),
          item("t1_t", "Tip: the Nathan vest doesn't bounce", author="tina"), item("t1_j", "lol", author="joe")]


class Stored(list):
    llm_up: list
    rebuilds: list


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(reddit.settings, "storage_path", tmp_path)
    monkeypatch.setattr(reddit, "feed_url", lambda: "https://www.reddit.com/user/me/saved.rss?feed=x&user=me")
    stored = Stored()
    monkeypatch.setattr("zettlecast.db.db.upsert_note", lambda note: stored.append(note))
    monkeypatch.setattr("zettlecast.linker.rebuild_graph", lambda: stored.rebuilds.append(1) or 0)
    stored.rebuilds = []
    llm_up = [True]

    def enrich(note):
        if llm_up[0]:
            note.metadata.tags, note.metadata.custom["summary"] = ["running"], "A summary."
        return note

    monkeypatch.setattr("zettlecast.enricher.enrich_note", enrich)
    monkeypatch.setattr("zettlecast.curation.curate", fake_curate(llm_up))
    stored.llm_up = llm_up
    return stored


def fake_curate(llm_up=(True,)):
    """Keeps the comments that give a tip; records what it was given."""
    from zettlecast.llm import LLMError

    def curate(title, subreddit, post, comments, saved=None):
        if not llm_up[0]:
            raise LLMError("down")
        curate.calls.append({"title": title, "post": post, "comments": comments, "saved": saved})
        return [{"n": i, "author": c["author"], "keep": "Tip" in c["text"], "kind": "product",
                 "note": c["text"].removeprefix("Tip: ")} for i, c in enumerate(comments, 1)]

    curate.calls = []
    return curate


def test_feed_link_must_be_a_private_saved_feed():
    with pytest.raises(reddit.RedditError):
        reddit.rss_feed("https://www.reddit.com/r/running/.rss")
    assert reddit.rss_feed("https://old.reddit.com/user/me/saved.json?feed=x&user=me") \
        == "https://www.reddit.com/user/me/saved.rss?feed=x&user=me"


def test_atom_entries_parse_like_reddit_sends_them():
    client = reddit.Client(interval=0)
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=FEED)))
    post, comment = client.get_feed("https://www.reddit.com/saved.rss")
    assert post["name"] == "t3_p1" and post["author"] == "op" and post["subreddit"] == "running"
    assert post["text"] == "Body with a source (https://example.com/a).\n\n- point"
    assert post["link"] == "https://example.com/study" and post["date"] == "2024-05-01T10:00:00"
    assert comment["name"] == "t1_c1" and comment["text"] == "A comment" and comment["link"] is None
    assert reddit._post_title(comment) == "Heat training study"


def test_feed_is_paged_until_a_short_page():
    first = [item(f"t3_{i}") for i in range(100)]
    pages = {None: first, "t3_99": [item("t1_c1")]}
    client = FakeClient({"saved.rss": lambda params: pages[params.get("after")]})
    items = reddit.fetch_saved(client, "https://www.reddit.com/saved.rss?feed=x")
    assert len(items) == 101 and items[-1]["name"] == "t1_c1"


def test_post_note_has_text_and_top_comments_but_no_bots_or_deleted():
    client = FakeClient({"/comments/p1/x/.rss": THREAD})
    note = reddit.build_note(client, item("t3_p1", "Body of the post."))
    assert client.calls[0][1]["depth"] == 1  # top-level comments only
    assert note.source_type == "reddit" and note.title == "Post t3_p1"
    assert note.uuid == reddit.note_uuid("t3_p1")  # stable: a refresh replaces the note
    text = note.full_text
    assert "## Post\nBody of the post." in text and "## Thread comments\n**u/alice**\nTop comment" in text
    assert "AutoModerator" not in text and "[deleted]" not in text
    assert [c["author"] for c in note.metadata.custom["reddit"]["comments"]] == ["alice", "tina", "joe"]
    assert note.metadata.custom["reddit"]["subreddit"] == "running"
    assert note.metadata.custom["reddit"]["thread_fetched"] is True


def test_link_post_reads_the_article_but_not_media(monkeypatch):
    read = []

    def fake_article(url):
        read.append(url)
        return {"url": url, "title": "The article", "text": "Article body."}

    monkeypatch.setattr(reddit, "_linked_article", fake_article)
    link = item("t3_p1", link="https://example.com/study")
    note = reddit.build_note(FakeClient({"/comments/p1/x/.rss": [link]}), link)
    assert "Link: https://example.com/study" in note.full_text
    assert "## Linked article: The article\nArticle body." in note.full_text
    assert read == ["https://example.com/study"]
    monkeypatch.undo()
    assert reddit._linked_article("https://i.redd.it/abc.jpg") is None
    assert reddit._linked_article("https://www.youtube.com/watch?v=x") is None


def test_saved_comment_note_has_the_post_and_other_comments():
    saved = item("t1_a", "Top comment", author="alice")
    client = FakeClient({"/comments/p1/x/.rss": THREAD + [item("t1_b", "Another view", author="bob")]})
    note = reddit.build_note(client, saved)
    assert client.calls[0][0] == "https://www.reddit.com/r/running/comments/p1/x/.rss"  # the post's thread
    assert note.title == "Comment by u/alice: Post t3_p1"
    text = note.full_text
    assert "## Saved comment\nTop comment" in text and "## The post\nBody of the post." in text
    assert "## Thread comments\n**u/tina**" in text and "**u/bob**\nAnother view" in text
    assert text.count("Top comment") == 1  # the saved comment isn't repeated among the others


def test_sync_imports_new_saves_once_and_refresh_retries_failed_threads(storage):
    feed = [item("t3_p1", "one"), item("t3_p2", "two")]
    feed[1]["permalink"] = "https://www.reddit.com/r/running/comments/p2/y/"
    client = FakeClient({"saved.rss": feed, "/comments/p1/x/.rss": THREAD,
                         "/comments/p2/y/.rss": reddit.RedditError("HTTP 403")})
    counts = reddit.sync(client=client)
    assert counts["imported"] == 2 and counts["partial"] == 1
    assert {n.title for n in storage} == {"Post t3_p1", "Post t3_p2"}

    storage.clear()
    assert reddit.sync(client=client)["imported"] == 0 and storage == []  # nothing new

    client.routes["/comments/p2/y/.rss"] = [feed[1]]
    counts = reddit.sync(client=client, refresh=True)
    assert counts["refreshed"] == 1 and [n.title for n in storage] == ["Post t3_p2"]
    assert reddit.status()["partial"] == 0


def test_sync_needs_a_feed(storage, monkeypatch):
    monkeypatch.setattr(reddit, "feed_url", lambda: "")
    with pytest.raises(reddit.RedditError, match="REDDIT_SAVED_FEED"):
        reddit.sync(client=FakeClient({}))


def test_client_waits_on_429_and_follows_rate_limit_headers():
    answers = [httpx.Response(429, headers={"retry-after": "7"}),
               httpx.Response(200, text=FEED, headers={"x-ratelimit-remaining": "5", "x-ratelimit-reset": "100"})]
    slept = []
    client = reddit.Client(interval=1, sleep=slept.append)
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: answers.pop(0)))
    assert len(client.get_feed("https://www.reddit.com/comments/x/.rss")) == 2
    assert 8.0 in slept
    # 5 requests left for 100 s: the next one waits ~20 s
    assert 19 < client.next_at - time.monotonic() <= 20


def test_each_subreddit_is_its_own_source_group():
    from zettlecast.linker import source_group

    note = reddit.build_note(None, item("t3_p1", "text"))
    assert source_group(note.model_dump()) == "reddit:r/running"


def test_feed_request_keeps_the_links_own_query():
    seen = []
    client = reddit.Client(interval=0, sleep=lambda s: None)
    client.http = httpx.Client(transport=httpx.MockTransport(
        lambda request: seen.append(dict(request.url.params)) or httpx.Response(200, text=FEED)))
    client.get_feed("https://www.reddit.com/saved.rss?feed=abc&user=me", {"limit": 100, "after": "t3_x"})
    assert seen == [{"feed": "abc", "user": "me", "limit": "100", "after": "t3_x"}]


def test_untagged_saves_are_tagged_later_from_the_database(storage, monkeypatch):
    by_uuid = {}
    monkeypatch.setattr("zettlecast.db.db.get_note_by_uuid", lambda uuid: by_uuid.get(uuid))
    client = FakeClient({"saved.rss": [item("t3_p1", "one"), item("t3_p2", "two")],
                         "/comments/p1/x/.rss": THREAD})
    counts = reddit.sync(client=client, enrich=False)  # GPU busy: no LLM
    assert counts["imported"] == 2 and counts["unenriched"] == 2
    assert storage[0].metadata.tags == [] and reddit.status()["untagged"] == 2
    by_uuid.update({n.uuid: n for n in storage})

    storage.llm_up[0] = False  # LLM still not answering: stops, nothing marked
    assert reddit.tag_untagged() == {"tagged": 0, "curated": 0, "failed": 1, "missing": 0, "stopped": False}

    storage.llm_up[0] = True
    del by_uuid[storage[1].uuid]  # deleted in the UI meanwhile
    calls = len(client.calls)
    counts = reddit.tag_untagged()
    assert counts["tagged"] == 1 and counts["missing"] == 1
    assert len(client.calls) == calls  # no Reddit requests
    assert storage[-1].metadata.tags == ["running"] and reddit.status()["untagged"] == 0


def test_a_sync_with_the_llm_tags_what_an_earlier_one_left_untagged(storage, monkeypatch):
    by_uuid = {}
    monkeypatch.setattr("zettlecast.db.db.get_note_by_uuid", lambda uuid: by_uuid.get(uuid))
    client = FakeClient({"saved.rss": [item("t3_p1", "one")], "/comments/p1/x/.rss": THREAD})
    reddit.sync(client=client, enrich=False)
    by_uuid.update({n.uuid: n for n in storage})
    counts = reddit.sync(client=client)
    assert counts["imported"] == 0 and counts["tagged"] == 1


def test_stop_request_ends_the_sync_after_the_current_save(storage):
    client = FakeClient({"saved.rss": [item(f"t3_p{i}", str(i)) for i in range(3)], "/comments/": THREAD})
    counts = reddit.sync(client=client, enrich=False, on_item=lambda n, total, title: reddit.request_stop())
    assert counts["imported"] == 1 and counts["stopped"]
    assert reddit.sync(client=client, enrich=False)["imported"] == 2  # the rest next time


def test_quick_sync_makes_no_request_per_save_and_comments_come_later_keeping_tags(storage, monkeypatch):
    by_uuid = {}
    monkeypatch.setattr("zettlecast.db.db.get_note_by_uuid", lambda uuid: by_uuid.get(uuid))
    client = FakeClient({"saved.rss": [item("t3_p1", "one")], "/comments/p1/x/.rss": THREAD})
    counts = reddit.sync(client=client, threads=False)
    assert counts["imported"] == 1 and counts["partial"] == 1
    assert [url for url, _ in client.calls if "comments" in url] == []
    tagged = storage[-1]
    assert "Top comment" not in tagged.full_text and tagged.metadata.tags == ["running"]
    by_uuid[tagged.uuid] = tagged

    counts = reddit.sync(client=client, refresh=True, enrich=False)  # fetch the comments, no LLM
    assert counts["refreshed"] == 1 and reddit.status()["partial"] == 0
    assert "Top comment" in storage[-1].full_text
    assert storage[-1].metadata.tags == ["running"] and storage[-1].metadata.custom["summary"] == "A summary."


def test_a_sync_links_its_notes_with_one_graph_rebuild(storage):
    client = FakeClient({"saved.rss": [item(f"t3_p{i}", str(i)) for i in range(3)], "/comments/": THREAD})
    assert reddit.sync(client=client, threads=False)["imported"] == 3
    assert storage.rebuilds == [1]
    reddit.sync(client=client, threads=False)  # nothing new: no rebuild
    assert storage.rebuilds == [1]


def test_takeaways_are_indexed_and_the_raw_comments_are_not(storage):
    from zettlecast.chunker import create_chunks

    client = FakeClient({"saved.rss": [item("t3_p1", "Body of the post.")], "/comments/p1/x/.rss": THREAD})
    reddit.sync(client=client)
    note = storage[-1]
    text = note.full_text
    assert "## From the comments\n- **Product**: the Nathan vest doesn't bounce (u/tina)" in text
    assert text.index("## From the comments") < text.index("## Thread comments")
    assert note.metadata.custom["reddit"]["curated"] and note.metadata.custom["reddit"]["takeaways"] == 1
    indexed = " ".join(c.text for c in create_chunks(text, "x"))
    assert "Nathan vest" in indexed and "lol" not in indexed and "Top comment" not in indexed
    assert reddit.status()["uncurated"] == 0


def test_a_saved_comment_is_the_reason_and_curation_sees_it(storage, monkeypatch):
    curate = fake_curate()
    monkeypatch.setattr("zettlecast.curation.curate", curate)
    saved = item("t1_a", "Top comment", author="alice")
    client = FakeClient({"saved.rss": [saved], "/comments/p1/x/.rss": THREAD})
    reddit.sync(client=client)
    call = curate.calls[0]
    assert call["saved"] == {"author": "alice", "text": "Top comment"}
    assert call["post"] == "Body of the post." and call["title"] == "Post t3_p1"
    assert "alice" not in [c["author"] for c in call["comments"]]
    assert "## Saved comment\nTop comment" in storage[-1].full_text


def test_notes_imported_before_curation_are_curated_from_the_database(storage, monkeypatch):
    """Older notes have their comments only in the text ("## Top comments"): moved and curated, tags redone."""
    from zettlecast.models import NoteMetadata, NoteModel

    old = NoteModel(uuid=reddit.note_uuid("t3_p1"), title="Post t3_p1", source_type="reddit", source_path="x",
                    full_text="r/running · posted by u/op\n\n## Post\nBody.\n\n## Top comments\n"
                              "**u/tina**\nTip: the Nathan vest doesn't bounce\n\n**u/joe**\nlol\n\nstill joe",
                    content_hash="x",
                    metadata=NoteMetadata(tags=["banter"], custom={"summary": "old", "reddit": {
                        "kind": "post", "subreddit": "running", "thread_fetched": True}}))
    monkeypatch.setattr("zettlecast.db.db.get_note_by_uuid", lambda uuid: old if uuid == old.uuid else None)
    reddit._save({"imported": {"t3_p1": {"uuid": old.uuid, "title": old.title, "thread": True, "enriched": True}},
                  "last_sync": None})
    assert reddit.status()["uncurated"] == 1
    counts = reddit.tag_untagged()
    assert counts["curated"] == 1 and counts["tagged"] == 1
    note = storage[-1]
    assert "## Top comments" not in note.full_text
    assert "## Thread comments\n**u/tina**" in note.full_text and "lol\n\nstill joe" in note.full_text
    assert "(u/tina)" in note.full_text
    assert note.metadata.tags == ["running"]  # redone from the curated text
    assert reddit.status()["uncurated"] == 0


def test_curation_parses_verdicts_and_drops_kept_comments_without_a_takeaway(monkeypatch):
    from zettlecast import curation

    monkeypatch.setattr(curation, "chat", lambda prompt, schema, temperature: {"comments": [
        {"n": 1, "keep": True, "kind": "advice", "note": "Run easy days slower."},
        {"n": 2, "keep": True, "kind": "joke", "note": "funny"},  # a drop kind can't be kept
        {"n": 3, "keep": True, "kind": "product", "note": ""}]})
    comments = [{"author": a, "text": "x"} for a in ("a", "b", "c", "d")]
    verdicts = curation.curate("T", "running", "", comments)
    assert [v["keep"] for v in verdicts] == [True, False, False, False]  # d: no verdict -> dropped
    assert curation.section(verdicts) == "## From the comments\n- **Advice**: Run easy days slower. (u/a)"


def test_rubric_is_the_users_once_saved(storage):
    from zettlecast import curation

    assert curation.rubric() == curation.DEFAULT_RUBRIC
    curation.save_rubric("Keep only shoe reviews.")
    assert curation.rubric() == "Keep only shoe reviews."
    curation.save_rubric("  ")
    assert curation.rubric() == curation.DEFAULT_RUBRIC
