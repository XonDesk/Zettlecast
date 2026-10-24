"""Screenshot triage: folders, decisions, undo, and what a kept screenshot's note indexes."""

import os
import time
from datetime import datetime, timedelta

import pytest
from PIL import Image

from zettlecast.chunker import create_chunks
from zettlecast.config import settings
from zettlecast.image import triage
from zettlecast.image.formatter import format_vision_result
from zettlecast.image.models import VisionExtraction
from zettlecast.image.queue import ImageQueue


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_path", tmp_path / "storage")
    shots = tmp_path / "shots"
    shots.mkdir()
    for i, colour in enumerate(["red", "green", "blue"]):
        path = shots / f"shot{i}.png"
        Image.new("RGB", (40, 30), colour).save(path)
        os.utime(path, (time.time() - 100 + i, time.time() - 100 + i))  # shot2 newest
    return shots


def test_folder_must_exist(storage):
    with pytest.raises(ValueError):
        triage.add_folder(str(storage / "missing"))
    for bad in ("", "  ", "shots"):  # would resolve against the server's working folder
        with pytest.raises(ValueError):
            triage.add_folder(bad)
    assert triage.add_folder(str(storage)) == [str(storage.resolve())]
    assert triage.add_folder(str(storage)) == [str(storage.resolve())]  # no duplicates


def test_pending_newest_first_and_decisions(storage):
    triage.add_folder(str(storage))
    result = triage.pending()
    assert [i["name"] for i in result["items"]] == ["shot2.png", "shot1.png", "shot0.png"]
    assert result["remaining"] == 3

    triage.decide(storage / "shot2.png", "skip")
    triage.decide(storage / "shot1.png", "keep")
    triage.decide(storage / "shot0.png", "keep_image")
    result = triage.pending()
    assert result["items"] == [] and result["remaining"] == 0
    assert result["decided"] == {"keep": 1, "keep_image": 1, "skip": 1, "ignore": 0}

    queue = ImageQueue()
    kept = {i.image.image_title: i.image.keep_image for i in queue.items.values()}
    assert kept == {"shot1": False, "shot0": True}  # skips are never queued


def test_ignore_adds_nothing_and_keeps_the_file(storage, monkeypatch):
    triage.add_folder(str(storage))
    triage.decide(storage / "shot1.png", "ignore", {"box": [0.1, 0.1, 0.5, 0.5], "points": []})
    assert ImageQueue().items == {}
    assert triage.pending()["decided"]["ignore"] == 1
    assert triage.pending()["remaining"] == 2  # not asked about again

    recycled = []
    monkeypatch.setattr(triage, "recycle", lambda path: recycled.append(path) or True)
    state = triage._load()
    triage._sweep(state, now=datetime.now() + timedelta(hours=1))
    assert recycled == [] and (storage / "shot1.png").exists()


def test_renamed_copy_is_not_asked_again(storage):
    triage.add_folder(str(storage))
    triage.decide(storage / "shot2.png", "skip")
    (storage / "shot2.png").rename(storage / "renamed.png")
    assert "renamed.png" not in [i["name"] for i in triage.pending()["items"]]


def test_undo_takes_a_keep_out_of_the_queue(storage):
    triage.add_folder(str(storage))
    triage.decide(storage / "shot1.png", "keep")
    assert ImageQueue().get_pending_count() == 1
    entry = triage.undo(storage / "shot1.png")
    assert entry["decision"] == "keep"
    assert ImageQueue().get_pending_count() == 0
    assert "shot1.png" in [i["name"] for i in triage.pending()["items"]]


def test_only_files_in_triage_folders(storage, tmp_path):
    triage.add_folder(str(storage))
    outside = tmp_path / "outside.png"
    Image.new("RGB", (4, 4)).save(outside)
    assert not triage.in_folders(outside)
    with pytest.raises(ValueError):
        triage.decide(outside, "keep")
    with pytest.raises(ValueError):
        triage.decide(storage / "shot0.png", "maybe")


def test_a_screenshot_gone_after_the_listing_is_skipped(storage, monkeypatch):
    # the Recycle Bin sweeper (or OneDrive) removed it between listing the folder and reading it
    triage.add_folder(str(storage))
    listed = triage._images_in(str(storage))
    (storage / "shot1.png").unlink()
    monkeypatch.setattr(triage, "_images_in", lambda folder: listed)
    p = triage.pending()
    assert [i["name"] for i in p["items"]] == ["shot2.png", "shot0.png"] and p["remaining"] == 2


def test_screenshot_note_indexes_only_its_focus():
    vision = VisionExtraction(
        kind="screenshot",
        focus="Heat acclimation takes about ten days of daily sessions to show up in plasma volume.",
        context="Reddit, r/ultrarunning: a reply by u/runner42 to a question about summer training.",
        title="Heat acclimation timeline",
        description="A Reddit comment explaining how long heat acclimation takes.",
        detected_text="Upvote 34\nShare\nOther comment about shoes and socks\nAd: buy a gravel bike",
        concepts=["heat acclimation"],
    )
    text = format_vision_result(vision, 1170, 2532, "PNG")
    indexed = " ".join(c.text for c in create_chunks(text, "n1"))
    assert "plasma volume" in indexed and "r/ultrarunning" in indexed
    assert "gravel bike" not in indexed and "Dimensions" not in indexed
    assert "gravel bike" in text  # kept in the note, just not embedded


THREAD = """01:35
r/trailrunning
Afraid I ran too little during taper
3 upvotes · 14 comments
CandidAd9050 · 17h
As others of said you're going to be fine. If you want more reassurance you should listen to the Science Of Ultra podcast about tapering.
Reply
diceclimber OP · 1h
Thanks for the reference. At first I read this as
the science of 'ultra-tapering'
@kevinmom
2 days ago
I do short, low intensity runs.
use.origin Sponsored
Get Origin for just $1 for 1 year
Add a comment"""


def test_screen_text_reads_like_the_thread():
    from zettlecast.image.formatter import screen_text

    focus = "As others of said you’re going to be fine. If you want more reassurance you should listen " \
            "to the Science Of Ultra podcast about tapering."
    assert screen_text(THREAD, focus) == (
        "r/trailrunning  \nAfraid I ran too little during taper\n\n"
        "**CandidAd9050**  \n_(the focus, above)_\n\n"
        "**diceclimber** (OP)  \nThanks for the reference. At first I read this as the science of 'ultra-tapering'\n\n"
        "**@kevinmom**  \nI do short, low intensity runs."
    )


def test_screen_text_author_on_its_own_line():
    from zettlecast.image.formatter import screen_text

    assert screen_text("runner42 OP\nTen days of heat.\n\nfast_feet\nAgreed.") == \
        "**runner42** (OP)  \nTen days of heat.\n\n**fast_feet**  \nAgreed."


def test_image_without_focus_indexes_its_text():
    vision = VisionExtraction(kind="photo", description="A race bib pinned to a shirt.",
                              detected_text="Western States 100 bib 12", concepts=["race bib"])
    text = format_vision_result(vision, 800, 600, "JPEG")
    indexed = " ".join(c.text for c in create_chunks(text, "n2"))
    assert "Western States 100 bib 12" in indexed


def test_names_outside_the_focus_are_dropped(monkeypatch):
    from zettlecast.image import vision_backend

    monkeypatch.setattr(vision_backend, "chat", lambda *a, **k: {
        "kind": "screenshot", "title": "Heat", "focus": "Julien Périard says ten days of heat sessions.",
        "context": "Reddit, a reply by u/heat_coach in r/ultrarunning", "description": "A comment.",
        "detected_text": "...", "concepts": ["heat"], "people": ["Julien Périard", "Gravel Guy"],
        "organisations": ["Reddit", "NorthPeak"]})
    image = __import__("pathlib").Path(__file__)  # any existing file; chat is mocked
    v = vision_backend.VisionBackend().analyze_image(image)
    assert v.people == ["Julien Périard"] and v.organisations == ["Reddit"]


def test_what_a_screenshot_was_kept_for_leads_the_note(monkeypatch):
    from zettlecast.image import vision_backend
    from zettlecast.image.formatter import saved_tag, saved_title

    monkeypatch.setattr(vision_backend, "chat", lambda *a, **k: {
        "saved_as": "media", "item": {"name": "Functional Programming in Scala",
                                      "by": "Paul Chiusano, Runar Bjarnason", "type": "book", "where": "", "price": ""},
        "kind": "screenshot", "title": "Book cover in a podcast video", "focus": "Functional Programming in Scala",
        "context": "The Peterman Pod on YouTube", "description": "A book cover.", "detected_text": "...",
        "concepts": ["functional programming"], "people": [], "organisations": []})
    v = vision_backend.VisionBackend().analyze_image(__import__("pathlib").Path(__file__))
    assert v.item == {"name": "Functional Programming in Scala", "by": "Paul Chiusano, Runar Bjarnason", "type": "book"}
    assert saved_tag(v) == "to read" and saved_title(v) == "Book: Functional Programming in Scala"
    text = format_vision_result(v, 1170, 2532, "PNG")
    assert text.startswith("## Kept\n\n**To read**: Functional Programming in Scala - Paul Chiusano")
    assert "To read" in " ".join(c.text for c in create_chunks(text, "n3"))


def test_places_and_products_keep_where_and_price():
    from zettlecast.image.formatter import saved_tag, saved_title

    place = VisionExtraction(saved_as="place", item={"name": "Kamikochi", "type": "valley", "where": "Nagano, Japan"},
                             title="Mountain valley", description="", detected_text="")
    assert saved_tag(place) == "place to visit" and saved_title(place) == "Place: Kamikochi"
    assert "- **Where**: Nagano, Japan" in format_vision_result(place, 10, 10)
    ring = VisionExtraction(saved_as="product", item={"name": "Blue stone ring", "where": "Amazon", "price": "$79.99"},
                            description="", detected_text="")
    assert saved_tag(ring) == "to buy" and "- **Price**: $79.99" in format_vision_result(ring, 10, 10)
    moment = VisionExtraction(saved_as="other", title="Sunset over the bay", description="", detected_text="")
    assert saved_tag(moment) == "" and saved_title(moment) == "Sunset over the bay"
    assert not format_vision_result(moment, 10, 10).startswith("## Kept")


def test_discards_go_to_the_recycle_bin_after_the_undo_window(storage, monkeypatch):
    from datetime import datetime

    recycled = []

    def fake_recycle(path):
        recycled.append(path.name)
        path.unlink()
        return True

    monkeypatch.setattr(triage, "recycle", fake_recycle)
    triage.add_folder(str(storage))
    triage.decide(storage / "shot0.png", "skip")
    triage.decide(storage / "shot1.png", "skip")
    triage.decide(storage / "shot2.png", "keep")  # keeps are never recycled

    triage.sweep()
    assert recycled == []  # still inside the undo window
    triage.undo(storage / "shot1.png")  # undone in time: stays

    later = datetime.now() + triage.RECYCLE_DELAY
    state = triage._load()
    assert triage._sweep(state, now=later)
    triage._save(state)
    assert recycled == ["shot0.png"]
    assert (storage / "shot1.png").exists() and (storage / "shot2.png").exists()

    entry = triage.undo(storage / "shot0.png")  # too late: it's in the Recycle Bin
    assert entry["recycled"]


def test_a_different_file_at_a_discarded_path_is_not_recycled(storage, monkeypatch):
    from datetime import datetime

    monkeypatch.setattr(triage, "recycle", lambda path: pytest.fail("must not recycle"))
    triage.add_folder(str(storage))
    triage.decide(storage / "shot0.png", "skip")
    Image.new("RGB", (40, 30), "white").save(storage / "shot0.png")  # replaced by a new screenshot
    os.utime(storage / "shot0.png", (time.time() + 5, time.time() + 5))
    state = triage._load()
    triage._sweep(state, now=datetime.now() + triage.RECYCLE_DELAY)
    assert next(iter(state["decisions"].values()))["recycled"] == "replaced"
    assert (storage / "shot0.png").exists()


def test_save_waits_for_a_reader_holding_the_state_file(tmp_path, monkeypatch):
    # Windows refuses to replace a file someone has open: the save must wait, not fail the request
    import threading

    monkeypatch.setattr(triage, "state_path", lambda: tmp_path / "image_triage.json")
    triage._save({"folders": [], "decisions": {}, "hashes": {}})
    reader = open(tmp_path / "image_triage.json", "rb")
    threading.Timer(0.2, reader.close).start()
    triage._save({"folders": ["x"], "decisions": {}, "hashes": {}})
    assert triage._load()["folders"] == ["x"]


def test_why_its_recommended_is_kept_with_the_thing():
    podcast = VisionExtraction(saved_as="media", item={"name": "The Science Of Ultra-Tapering", "by": "Science Of Ultra",
                                                       "type": "podcast", "why": "Running less before an ultra is normal."},
                               description="", detected_text="")
    text = format_vision_result(podcast, 10, 10)
    assert text.startswith("## Kept\n\n**To listen**: The Science Of Ultra-Tapering - Science Of Ultra\n"
                           "- **Why**: Running less before an ultra is normal.")


def test_reprocess_puts_back_only_the_chosen_notes_images(storage):
    from zettlecast.image.image_parser import image_note_uuid

    triage.add_folder(str(storage))
    triage.decide(storage / "shot1.png", "keep")
    triage.decide(storage / "shot0.png", "keep_image")
    queue = ImageQueue()
    for job in queue.items.values():
        queue.mark_completed(job.image.id, 1.0)
    assert ImageQueue().reprocess({image_note_uuid(storage / "shot1.png")}) == 1
    pending = [j for j in ImageQueue().items.values() if j.status == "pending"]
    assert [(j.image.image_title, j.image.keep_image) for j in pending] == [("shot1", False)]  # triage choice kept
    assert ImageQueue().reprocess() == 1  # the rest
