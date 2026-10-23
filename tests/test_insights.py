"""Today's insight cards: which links are chosen, the day's set, verdicts, the LLM's part."""

import json
from datetime import datetime, timedelta

import pytest

from zettlecast import insights


def note(uuid, source_type="audio", group=None, community=0, date="2022-01-01", title=None):
    return {"uuid": uuid, "title": title or f"Note {uuid}", "source_type": source_type,
            "group": group or f"{source_type}:x", "community": community, "community_label": f"c{community}",
            "date": datetime.fromisoformat(date), "brief": f"About {uuid}.", "image": False}


def edge(a, b, weight=0.5, passage=True, kind="similar", vector_sim=0.5, entity_sim=0.0, shared=()):
    reason = {"shared": [{"kind": k, "name": n} for k, n in shared]}
    if passage:
        reason["passage"] = {"similarity": 0.6, "source": {"text": f"[1.0s - 2.0s] {a} says", "time": 1.0},
                             "target": {"text": f"## Summary\n{b} **says**", "time": None}}
    return {"source_uuid": a, "target_uuid": b, "weight": weight, "kind": kind, "vector_sim": vector_sim,
            "passage_sim": 0.6 if passage else 0.0, "entity_sim": entity_sim, "reason_json": json.dumps(reason)}


@pytest.fixture
def graph(tmp_path, monkeypatch):
    """Storage in tmp; notes and edges the test sets."""
    monkeypatch.setattr(insights.settings, "storage_path", tmp_path)
    state = {"notes": {}, "edges": []}
    monkeypatch.setattr(insights, "_notes", lambda: state["notes"])
    monkeypatch.setattr("zettlecast.db.db.get_all_edges", lambda: state["edges"])
    monkeypatch.setattr("zettlecast.db.db.note_titles", lambda uuids: {u: ("t", "") for u in uuids if u in state["notes"]})
    rejected = []
    monkeypatch.setattr("zettlecast.db.db.add_rejected_edge", lambda a, b: rejected.append((a, b)))
    state["rejected"] = rejected
    monkeypatch.setattr("zettlecast.runstate.active_run", lambda: None)
    return state


def test_a_link_across_sources_and_communities_beats_the_same_guest_on_the_same_show():
    a, b = note("a", "audio", "podcast:Koop", 0), note("b", "audio", "podcast:Koop", 1)
    screenshot = note("c", "image", "image", 2)
    same_show = insights.surprise(edge("a", "b", entity_sim=0.3), a, b, shared_people=1)
    across = insights.surprise(edge("a", "c"), a, screenshot)
    assert across > same_show


def test_candidates_skip_duplicates_near_copies_and_the_weaker_half():
    notes = {u: note(u, community=i) for i, u in enumerate("abcdef")}
    edges = [edge("a", "b", 0.6), edge("c", "d", 0.1), edge("e", "f", 0.7, kind="duplicate"),
             edge("a", "c", 0.8, vector_sim=0.95), edge("b", "d", 0.5)]
    keys = [c["key"] for c in insights.candidates(notes, edges, exclude=set())]
    assert insights.pair_key("a", "b") in keys
    assert insights.pair_key("c", "d") not in keys  # weaker half
    assert insights.pair_key("e", "f") not in keys  # duplicate
    assert insights.pair_key("a", "c") not in keys  # near-copy


def test_an_echo_is_something_recent_with_something_much_older():
    now = datetime(2026, 10, 22)
    notes = {"new": note("new", "image", date="2026-10-20"), "old": note("old", date="2023-01-01"),
             "mid": note("mid", date="2026-09-01")}
    out = insights.candidates(notes, [edge("new", "old", 0.5), edge("new", "mid", 0.5)], set(), now=now)
    kinds = {c["key"]: c["kind"] for c in out}
    assert kinds[insights.pair_key("new", "old")] == "echo"
    assert kinds[insights.pair_key("new", "mid")] == "surprise"  # not old enough
    echo = next(c for c in out if c["kind"] == "echo")
    assert echo["a"]["uuid"] == "new"


def test_the_day_has_a_few_cards_no_note_twice_and_no_repeat_the_next_day(graph):
    graph["notes"] = {u: note(u, community=i % 3, source_type=("audio", "image", "reddit")[i % 3])
                      for i, u in enumerate("abcdefghijklmnop")}
    graph["edges"] = [edge(x, y, 0.5 + i / 100) for i, (x, y) in enumerate(zip("abcdefghijklmno", "bcdefghijklmnop"))]
    first = insights.today()
    assert 1 <= len(first["cards"]) <= insights.PER_DAY
    ids = [s["uuid"] for c in first["cards"] for s in (c["a"], c["b"])]
    assert len(ids) == len(set(ids))
    card = first["cards"][0]
    assert card["a"]["text"] and "[" not in card["a"]["text"][:1] and "**" not in card["b"]["text"]

    data = insights._load()
    tomorrow = insights.plan_day(data, (datetime.now() + timedelta(days=1)).date().isoformat())
    assert not set(tomorrow) & {c["key"] for c in first["cards"]}
    assert insights.today()["cards"] == first["cards"]  # the same set all day


def test_prepared_cards_go_first_and_unrelated_ones_are_never_shown(graph):
    graph["notes"] = {u: note(u, community=i) for i, u in enumerate("abcdefghijklmnop")}
    graph["edges"] = [edge("a", "b", 0.9), edge("c", "d", 0.8), edge("e", "f", 0.7), edge("g", "h", 0.6)]
    # weak links elsewhere, so the four above are the stronger half
    graph["edges"] += [edge(x, y, 0.1) for x, y in zip("ikmo", "jlnp")]
    data = insights._load()
    insights.refill(data)
    weakest = insights.pair_key("g", "h")
    strongest = insights.pair_key("a", "b")
    data["cards"][weakest]["insight"] = {"relation": "extends", "insight": "x", "question": "q", "note_title": "t"}
    data["cards"][strongest]["insight"] = {"relation": "unrelated", "insight": "x", "question": "q", "note_title": "t"}
    plan = insights.plan_day(data, "2026-10-22")
    assert plan[0] == weakest
    assert strongest not in plan


def test_not_related_unlinks_the_pair_for_good(graph):
    graph["notes"] = {u: note(u, community=i) for i, u in enumerate("ab")}
    graph["edges"] = [edge("a", "b")]
    key = insights.today()["cards"][0]["key"]
    insights.decide(key, "rejected")
    assert graph["rejected"] == [("a", "b")]
    assert insights._load()["cards"][key]["verdict"] == "rejected"


def test_the_worker_claims_cards_and_sends_back_the_llms_take(graph):
    graph["notes"] = {u: note(u, community=i) for i, u in enumerate("abcd")}
    graph["edges"] = [edge("a", "b"), edge("c", "d")]
    insights.today()
    assert insights.pending_count() == 2
    claimed = insights.claim(10, 600)
    assert len(claimed) == 2 and insights.claim(10, 600) == []  # held while the worker has them
    take = {"relation": "contradicts", "insight": "A says x, B says not x.", "question": "Which?", "note_title": "X"}
    insights.store(claimed[0]["key"], take)
    insights.store(claimed[1]["key"], error="LLM down")
    assert insights._load()["cards"][claimed[0]["key"]]["insight"] == take
    assert insights.pending_count() == 1  # the failed one is tried again
    assert any(c["insight"] == take for c in insights.today()["cards"])


def test_generate_asks_for_the_relation_and_rejects_an_unusable_answer(monkeypatch):
    from zettlecast.llm import LLMError

    card = {"a": {"title": "A", "source_type": "audio", "date": "2022-01-01", "text": "x", "passage": True},
            "b": {"title": "B", "source_type": "image", "date": None, "text": "y", "passage": False},
            "shared": ["iron"]}
    asked = {}

    def chat(prompt, schema, system, temperature):
        asked["prompt"] = prompt
        return {"relation": "applies", "insight": " B is A in practice. ", "question": "How?", "note_title": "T"}

    monkeypatch.setattr("zettlecast.llm.chat", chat)
    assert insights.generate(card)["insight"] == "B is A in practice."
    assert "podcast" in asked["prompt"] and "screenshot" in asked["prompt"] and "iron" in asked["prompt"]
    monkeypatch.setattr("zettlecast.llm.chat", lambda *a, **k: {"relation": "maybe", "insight": ""})
    with pytest.raises(LLMError):
        insights.generate(card)


def test_a_note_from_a_card_links_both_notes(graph, monkeypatch):
    graph["notes"] = {"a": note("a", title="Iron and running"), "b": note("b", community=1, title="Ferritin talk")}
    graph["edges"] = [edge("a", "b")]
    key = insights.today()["cards"][0]["key"]
    stored = []
    monkeypatch.setattr("zettlecast.pipeline.ingest_note", lambda n: stored.append(n))
    uuid = insights.write_note(key, "Low ferritin hides behind fatigue", "My take.")
    text = stored[0].full_text
    assert "My take." in text and "[[Iron and running]]" in text and "[[Ferritin talk]]" in text
    assert stored[0].uuid == uuid
    assert (insights.settings.storage_path / "notes").glob("*.md")
    assert insights._load()["cards"][key]["verdict"] == "noted"


def test_a_card_found_unrelated_is_replaced_on_the_day(graph):
    ids = [f"n{i}" for i in range(40)]
    graph["notes"] = {u: note(u, community=i, source_type=("audio", "image")[i % 2]) for i, u in enumerate(ids)}
    # ten strong links (two days' worth) and ten weak ones below them
    graph["edges"] = [edge(ids[2 * i], ids[2 * i + 1], 0.5 + i / 100 if i < 10 else 0.1) for i in range(20)]
    first = insights.today()["cards"]
    assert len(first) == insights.PER_DAY
    insights.store(first[0]["key"], {"relation": "unrelated", "insight": "x", "question": "q", "note_title": "t"})
    again = insights.today()["cards"]
    assert len(again) == insights.PER_DAY
    assert first[0]["key"] not in {c["key"] for c in again}
    assert [c["key"] for c in again[:4]] == [c["key"] for c in first[1:]]  # the rest stay, in order
