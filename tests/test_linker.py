"""Tests for graph edge construction."""

from unittest.mock import patch

import numpy as np
import pytest

from zettlecast.linker import _edges_from, calculate_composite_weight, calculate_jaccard_similarity


class TestJaccardSimilarity:
    def test_identical_tags(self):
        assert calculate_jaccard_similarity(["a", "b"], ["a", "b"]) == 1.0

    def test_no_overlap(self):
        assert calculate_jaccard_similarity(["a"], ["b"]) == 0.0

    def test_partial_overlap(self):
        assert calculate_jaccard_similarity(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(0.5)

    def test_empty_tags(self):
        assert calculate_jaccard_similarity([], []) == 0.0

    def test_case_and_whitespace_insensitive(self):
        assert calculate_jaccard_similarity(["Gut Training "], ["gut training"]) == 1.0


class TestCompositeWeight:
    def test_pure_vector_sim(self):
        assert calculate_composite_weight(0.8, 0.0, alpha=1.0, beta=0.0) == 0.8

    def test_pure_tag_sim(self):
        assert calculate_composite_weight(0.0, 0.6, alpha=0.0, beta=1.0) == 0.6

    def test_negative_vector_sim_clamped(self):
        assert calculate_composite_weight(-0.5, 0.0, alpha=1.0, beta=0.0) == 0.0

    def test_weights_normalized(self):
        assert calculate_composite_weight(1.0, 1.0, alpha=2.0, beta=2.0) == 1.0


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def note(uuid, vector, created, tags=()):
    return {"uuid": uuid, "title": uuid, "created_at": created, "tags": list(tags), "vector": unit(vector)}


@pytest.fixture(autouse=True)
def graph_settings():
    with patch("zettlecast.linker.settings") as s:
        s.graph_alpha = 0.7
        s.graph_beta = 0.3
        s.graph_edge_threshold = 0.5
        s.graph_max_edges_per_note = 8
        yield s


class TestEdgesFrom:
    def test_links_similar_notes_older_to_newer(self):
        a = note("a", [1, 0.1], "2026-02-01", ["gut"])
        b = note("b", [1, 0.0], "2026-01-01", ["gut"])
        edges = _edges_from(a, [a, b], rejected=set())
        assert len(edges) == 1
        assert (edges[0]["source_uuid"], edges[0]["target_uuid"]) == ("b", "a")
        assert edges[0]["tag_sim"] == 1.0

    def test_skips_dissimilar_and_self(self):
        a = note("a", [1, 0], "2026-01-01")
        c = note("c", [0, 1], "2026-01-02")
        assert _edges_from(a, [a, c], rejected=set()) == []

    def test_respects_rejected_pairs(self):
        a = note("a", [1, 0], "2026-01-01")
        b = note("b", [1, 0], "2026-01-02")
        assert _edges_from(a, [a, b], rejected={("a", "b")}) == []

    def test_caps_edges_per_note(self, graph_settings):
        graph_settings.graph_max_edges_per_note = 2
        a = note("a", [1, 0], "2026-01-01")
        others = [note(f"n{i}", [1, 0.01 * i], f"2026-01-0{i + 2}") for i in range(5)]
        edges = _edges_from(a, [a, *others], rejected=set())
        assert len(edges) == 2
        assert edges[0]["weight"] >= edges[1]["weight"]
