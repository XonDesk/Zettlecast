"""Tests for graph edges and communities (synthetic vectors, no database)."""

from unittest.mock import patch

import numpy as np
import pytest

from zettlecast import linker
from zettlecast.config import settings

DIM = settings.embedding_dimensions


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def axis(*parts):
    """A unit vector mixing the given axes."""
    v = np.zeros(DIM, np.float32)
    for a in parts:
        v[a] += 1
    return unit(v)


def note(uuid, vector, created="2026-01-01", show="Show", tags=(), people=(), orgs=(), text="", source_type="audio"):
    custom = {"podcast": {"show": show}} if show else {}
    if people:
        custom["people"] = list(people)
    if orgs:
        custom["organisations"] = list(orgs)
    return {"uuid": uuid, "title": uuid, "source_type": source_type, "source_path": f"/{uuid}",
            "created_at": created, "metadata": {"tags": list(tags), "custom": custom},
            "full_text": text, "vector": vector}


def chunk(note_uuid, vector, text="text"):
    return {"note_uuid": note_uuid, "text": text, "vector": vector}


@pytest.fixture(autouse=True)
def graph_settings(monkeypatch):
    for name, value in {"graph_weight_semantic": 0.5, "graph_weight_passage": 0.3, "graph_weight_entity": 0.2,
                        "graph_max_edges_per_note": 2, "graph_min_weight": 0.05,
                        "graph_passage_similarity": 0.8, "graph_boilerplate_similarity": 0.9,
                        "graph_boilerplate_share": 0.2, "graph_duplicate_share": 0.9,
                        "graph_resolution": 1.0}.items():
        monkeypatch.setattr(settings, name, value)


class TestSourceGroups:
    def test_show_domain_type(self):
        assert linker.source_group(note("a", None, show="KoopCast")) == "podcast:KoopCast"
        web = note("b", None, show=None, source_type="web")
        web["metadata"]["source_url"] = "https://www.example.com/x"
        assert linker.source_group(web) == "web:example.com"
        assert linker.source_group(note("c", None, show=None, source_type="image")) == "image"

    def test_balanced_centre_counts_each_group_once(self):
        # ten notes of a big show around axis 0, one web note on axis 1: the subtracted mean is
        # halfway between the two groups, not (10 a + b) / 11
        big = [unit(axis(0) + 0.1 * axis(10 + k)) for k in range(10)]
        vectors = np.stack(big + [axis(1)])
        centred = linker.balanced_centre(vectors, ["podcast:A"] * 10 + ["web:b"])
        mean = (np.stack(big).mean(axis=0) + axis(1)) / 2
        assert np.allclose(centred[10], unit(axis(1) - mean), atol=1e-6)

    def test_single_group_loses_what_its_notes_share(self):
        big = [unit(axis(0) + 0.3 * axis(10 + k)) for k in range(5)]
        centred = linker.balanced_centre(np.stack(big), ["podcast:A"] * 5)
        assert centred[0] @ centred[1] < 0  # only their differences are left


class TestEntities:
    def test_topic_folding(self):
        assert linker.topic_key("Ultra running") == linker.topic_key("ultrarunning") == linker.topic_key("ultra-running")
        assert linker.topic_key("Intervals") == linker.topic_key("interval")
        assert linker.topic_key("fitness") == "fitness"

    def test_people_spellings_merge_but_not_different_people(self):
        keys = linker._people_keys(["Olaf Alexander Bu", "Olav Aleksander Bu", "John Smith", "Jane Smith"])
        assert keys["Olaf Alexander Bu"] == keys["Olav Aleksander Bu"]
        assert keys["John Smith"] != keys["Jane Smith"]

    def test_host_in_every_note_adds_nothing_a_shared_guest_does(self):
        notes = [note(f"n{i}", axis(i), people=["Jason Koop"] + (["Sophie Herzog"] if i < 2 else [])) for i in range(6)]
        idx = linker.build_entities(notes)
        assert idx.similarity[0, 1] > 0.5  # same guest
        assert idx.similarity[2, 3] == 0  # only the host in common
        assert idx.per_note[0]["person:jason koop"] == pytest.approx(0)


class TestPassages:
    def test_shares_and_boilerplate(self):
        intro = axis(0)
        # four notes of one show, each opening with the same intro; notes 0 and 1 also share a topic passage
        topic = axis(1)
        chunks = []
        for i in range(4):
            chunks.append(chunk(f"n{i}", intro, "[0.0s] Host: welcome"))
            chunks.append(chunk(f"n{i}", topic if i < 2 else axis(10 + i), f"[{60 * i}.0s] Guest: body"))
        notes = [note(f"n{i}", axis(20 + i)) for i in range(4)]
        passages = linker.load_passages(notes, chunks)
        shares = linker.passage_shares(passages, ["podcast:Show"] * 4)
        assert passages.boilerplate[0] and not passages.boilerplate[1]  # intro skipped, body kept
        assert shares[0, 1] == pytest.approx(1.0)  # its only kept passage matches
        assert shares[0, 2] == 0

    def test_focus_matches_full_computation(self):
        rng = np.random.default_rng(0)
        base = [unit(rng.normal(size=DIM)) for _ in range(3)]
        chunks = [chunk(f"n{i}", unit(base[(i + k) % 3] + 0.3 * unit(rng.normal(size=DIM)))) for i in range(5) for k in range(3)]
        notes = [note(f"n{i}", axis(i)) for i in range(5)]
        groups = [f"g{i}" for i in range(5)]  # one note per group: no boilerplate either way
        with patch.object(settings, "graph_passage_similarity", 0.85):
            full = linker.passage_shares(linker.load_passages(notes, chunks), groups)
            focus = linker.passage_shares(linker.load_passages(notes, chunks), groups, focus=2)
        assert np.allclose(full[2], focus[2]) and np.allclose(full[:, 2], focus[:, 2])

    def test_best_pair_reports_times(self):
        notes = [note("a", axis(0)), note("b", axis(1))]
        chunks = [chunk("a", axis(5), "[12.5s] A: heat training"), chunk("b", axis(5), "[99.0s - 101.0s] B: heat")]
        pair = linker.best_pair(linker.load_passages(notes, chunks), 0, 1)
        assert pair["source"]["time"] == 12.5 and pair["target"]["time"] == 99.0


class TestEdges:
    def corpus(self):
        # two topics, three notes each; inside a topic the notes are alike
        notes = [note(f"t{t}n{k}", unit(axis(t) + 0.2 * axis(10 + 3 * t + k)), created=f"2026-01-0{k + 1}",
                      tags=[f"topic {t}"]) for t in range(2) for k in range(3)]
        return notes, [chunk(n["uuid"], n["vector"]) for n in notes]

    def test_k_best_union_stays_inside_topics(self):
        notes, chunks = self.corpus()
        graph = linker.build(notes, chunks)
        pairs = linker.select_pairs(graph, set())
        assert pairs and all(notes[i]["uuid"][:2] == notes[j]["uuid"][:2] for i, j in pairs)

    def test_rejected_pair_never_linked(self):
        notes, chunks = self.corpus()
        graph = linker.build(notes, chunks)
        pairs = linker.select_pairs(graph, {("t0n0", "t0n1")})
        assert (0, 1) not in pairs

    def test_wikilink_is_an_edge_of_its_own_kind(self):
        notes, chunks = self.corpus()
        notes[0]["full_text"] = "See [[t1n2]] and [[Missing note]]"
        graph = linker.build(notes, chunks)
        assert graph.links == {(0, 5)}
        assert (0, 5) in linker.select_pairs(graph, set())
        edge = linker._edge(graph, 0, 5, linker._kind(graph, 0, 5))
        assert edge["kind"] == "link" and edge["weight"] == 1.0

    def test_edges_point_from_older_to_newer(self):
        notes, chunks = self.corpus()
        edge = linker._edge(linker.build(notes, chunks), 1, 0, "similar")
        assert (edge["source_uuid"], edge["target_uuid"]) == ("t0n0", "t0n1")


class TestCommunities:
    def test_two_topics_two_communities_with_stable_ids(self):
        notes = [note(f"t{t}n{k}", unit(axis(t) + 0.2 * axis(10 + 3 * t + k)), tags=[f"topic {t}", "common"])
                 for t in range(2) for k in range(4)]
        graph = linker.build(notes, [chunk(n["uuid"], n["vector"]) for n in notes])
        edges = [linker._edge(graph, i, j, "similar") for i, j in linker.select_pairs(graph, set())]
        first = linker.detect_communities(graph, edges, [])
        assert sorted(len(c["members"]) for c in first) == [4, 4]
        assert {c["label"] for c in first} == {"topic 0 · common", "topic 1 · common"}
        # a rebuild with the ids swapped in storage keeps each group's id
        previous = [{"community": 7 if c["members"][0].startswith("t0") else 3, "members": c["members"]} for c in first]
        again = {c["members"][0][:2]: c["community"] for c in linker.detect_communities(graph, edges, previous)}
        assert again == {"t0": 7, "t1": 3}
