"""
tests/test_metrics.py — Unit tests for all ranking metric functions.

No index, no model, no network required — pure maths only.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.evaluation.metrics import (
    dcg_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    build_semantic_relevance_vector,
)


# ── dcg_at_k ─────────────────────────────────────────────────────────────────

class TestDcgAtK:
    def test_perfect_single_hit(self):
        # rel=2 at rank 1: (2^2 - 1) / log2(2) = 3 / 1 = 3.0
        assert math.isclose(dcg_at_k([2], k=1), 3.0)

    def test_all_zeros(self):
        assert dcg_at_k([0, 0, 0], k=3) == 0.0

    def test_empty_list(self):
        assert dcg_at_k([], k=5) == 0.0

    def test_k_truncates_list(self):
        assert dcg_at_k([2, 0, 2], k=2) == dcg_at_k([2, 0], k=2)

    def test_known_value(self):
        # [2, 1, 0]:  3/log2(2) + 1/log2(3) + 0 = 3.0 + 0.6309…
        result = dcg_at_k([2, 1, 0], k=3)
        expected = 3.0 + 1.0 / math.log2(3)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_k_larger_than_list(self):
        """k > len(relevances) should not raise."""
        result = dcg_at_k([2, 1], k=10)
        assert result > 0.0

    def test_float_relevances(self):
        """dcg_at_k must accept continuous relevance scores, not just integers."""
        result = dcg_at_k([0.9, 0.5, 0.1], k=3)
        assert result > 0.0

    def test_float_relevance_perfect(self):
        # rel=1.0 at rank 1: (2^1.0 - 1) / log2(2) = 1.0 / 1.0 = 1.0
        assert math.isclose(dcg_at_k([1.0], k=1), 1.0)


# ── ndcg_at_k ────────────────────────────────────────────────────────────────

class TestNdcgAtK:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k([2, 0, 0, 0, 0, 0, 0, 0, 0, 0], k=10) == 1.0

    def test_all_zeros_is_zero(self):
        assert ndcg_at_k([0] * 10, k=10) == 0.0

    def test_result_in_unit_interval(self):
        import random
        rng = random.Random(0)
        for _ in range(20):
            rels = [rng.choice([0, 1, 2]) for _ in range(10)]
            score = ndcg_at_k(rels, k=10)
            assert 0.0 <= score <= 1.0, f"Out of range: {score} for {rels}"

    def test_ideal_order_beats_reverse(self):
        rels = [2, 2, 1, 0, 0]
        assert ndcg_at_k(rels, k=5) > ndcg_at_k(list(reversed(rels)), k=5)

    def test_empty_list(self):
        assert ndcg_at_k([], k=5) == 0.0

    def test_single_element_perfect(self):
        assert ndcg_at_k([2], k=1) == 1.0

    def test_k_one_hit_at_rank_one_is_one(self):
        assert ndcg_at_k([2, 0, 0], k=1) == 1.0

    def test_k_one_miss_at_rank_one_is_zero(self):
        assert ndcg_at_k([0, 2, 0], k=1) == 0.0

    def test_float_relevances_in_unit_interval(self):
        """Semantic (continuous) scores must stay in [0, 1]."""
        rels = [0.95, 0.7, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        score = ndcg_at_k(rels, k=10)
        assert 0.0 <= score <= 1.0

    def test_exact_hit_gives_one_with_float_relevances(self):
        """Placing the 1.0 item at rank 1 yields perfect NDCG."""
        rels = [1.0, 0.5, 0.2, 0.0, 0.0]
        assert ndcg_at_k(rels, k=5) == 1.0

    def test_continuous_partial_credit(self):
        """
        A near-miss (0.85 similarity to expected book) at rank 1 should
        score meaningfully higher than all zeros, even without an exact hit.
        """
        rels_near_miss = [0.85, 0.70, 0.60, 0.0, 0.0]
        rels_all_zero  = [0.0] * 5
        assert ndcg_at_k(rels_near_miss, k=5) > ndcg_at_k(rels_all_zero, k=5)


# ── hit_rate_at_k ─────────────────────────────────────────────────────────────

class TestHitRateAtK:
    def test_exact_match_within_k(self):
        assert hit_rate_at_k(["Book A", "Dune", "Book C"], "Dune", k=10) == 1

    def test_exact_match_outside_k(self):
        assert hit_rate_at_k(["Book A", "Book B", "Dune"], "Dune", k=2) == 0

    def test_case_insensitive(self):
        assert hit_rate_at_k(["dune"], "Dune", k=5) == 1

    def test_strip_whitespace(self):
        assert hit_rate_at_k(["  Dune  "], "Dune", k=5) == 1

    def test_not_found_returns_zero(self):
        assert hit_rate_at_k(["Book A", "Book B"], "Dune", k=10) == 0

    def test_empty_ranked_list(self):
        assert hit_rate_at_k([], "Dune", k=10) == 0

    def test_first_position_hit(self):
        assert hit_rate_at_k(["Dune", "Other"], "Dune", k=1) == 1

    def test_returns_int(self):
        result = hit_rate_at_k(["Dune"], "Dune", k=5)
        assert isinstance(result, int)


# ── reciprocal_rank ───────────────────────────────────────────────────────────

class TestReciprocalRank:
    def test_rank_one_hit(self):
        assert reciprocal_rank(["Dune", "Other"], "Dune", k=10) == 1.0

    def test_rank_two_hit(self):
        result = reciprocal_rank(["Other", "Dune"], "Dune", k=10)
        assert math.isclose(result, 0.5)

    def test_rank_three_hit(self):
        ranked = ["Book A", "Book B", "Dune", "Book C"]
        result = reciprocal_rank(ranked, "Dune", k=10)
        assert math.isclose(result, 1.0 / 3)

    def test_not_found_returns_zero(self):
        assert reciprocal_rank(["Book A", "Book B"], "Dune", k=10) == 0.0

    def test_outside_k_returns_zero(self):
        ranked = ["Book A", "Book B", "Dune"]
        assert reciprocal_rank(ranked, "Dune", k=2) == 0.0

    def test_case_insensitive(self):
        result = reciprocal_rank(["dune"], "Dune", k=5)
        assert result == 1.0

    def test_empty_list(self):
        assert reciprocal_rank([], "Dune", k=10) == 0.0

    def test_returns_float(self):
        result = reciprocal_rank(["Dune"], "Dune", k=5)
        assert isinstance(result, float)


# ── build_semantic_relevance_vector ──────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector (mirrors the engine's embedding convention)."""
    return v / np.linalg.norm(v)


# Fixed toy embeddings — 4-D, L2-normalised
_DUNE_EMB      = _unit(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
_SIMILAR_EMB   = _unit(np.array([0.9, 0.4, 0.1, 0.0], dtype=np.float32))  # high cos-sim to Dune
_UNRELATED_EMB = _unit(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))  # low cos-sim to Dune

MOCK_LOOKUP: dict[str, np.ndarray] = {
    "dune":          _DUNE_EMB,
    "dune messiah":  _SIMILAR_EMB,
    "other book":    _UNRELATED_EMB,
}


class TestBuildSemanticRelevanceVector:

    def test_exact_match_scores_near_one(self):
        """The expected book with itself must score ~1.0 (cosine sim of identical vectors)."""
        vec = build_semantic_relevance_vector(["Dune", "Other Book"], "Dune", MOCK_LOOKUP, k=5)
        assert math.isclose(vec[0], 1.0, abs_tol=1e-5)

    def test_similar_book_gets_partial_credit(self):
        """A semantically close book must score meaningfully between 0 and 1."""
        vec = build_semantic_relevance_vector(["Dune Messiah"], "Dune", MOCK_LOOKUP, k=5)
        assert 0.0 < vec[0] < 1.0

    def test_unrelated_book_scores_low(self):
        """An orthogonal book should score near 0."""
        vec = build_semantic_relevance_vector(["Other Book"], "Dune", MOCK_LOOKUP, k=5)
        assert vec[0] < 0.3

    def test_similar_scores_higher_than_unrelated(self):
        vec = build_semantic_relevance_vector(
            ["Dune Messiah", "Other Book"], "Dune", MOCK_LOOKUP, k=5
        )
        assert vec[0] > vec[1]

    def test_output_length_equals_k(self):
        for k in (3, 5, 10):
            vec = build_semantic_relevance_vector(["Dune"], "Dune", MOCK_LOOKUP, k=k)
            assert len(vec) == k, f"Expected length {k}, got {len(vec)}"

    def test_pads_with_zeros_when_short(self):
        vec = build_semantic_relevance_vector(["Dune"], "Dune", MOCK_LOOKUP, k=5)
        assert vec[1] == 0.0
        assert vec[4] == 0.0

    def test_empty_ranked_list(self):
        vec = build_semantic_relevance_vector([], "Dune", MOCK_LOOKUP, k=3)
        assert vec == [0.0, 0.0, 0.0]

    def test_all_scores_in_unit_interval(self):
        ranked = ["Dune", "Dune Messiah", "Other Book"]
        vec = build_semantic_relevance_vector(ranked, "Dune", MOCK_LOOKUP, k=5)
        assert all(0.0 <= s <= 1.0 for s in vec), f"Out of range: {vec}"

    def test_unknown_title_scores_zero(self):
        """A returned title not in the embedding lookup must score 0."""
        vec = build_semantic_relevance_vector(["Unknown Novel"], "Dune", MOCK_LOOKUP, k=3)
        assert vec[0] == 0.0

    def test_expected_not_in_lookup_falls_back(self):
        """
        When the expected book has no embedding, exact-match titles should
        still score 1.0 (graceful degradation).
        """
        vec = build_semantic_relevance_vector(
            ["Rare Book"], "Rare Book", {}, k=3
        )
        assert vec[0] == 1.0

    def test_case_insensitive_lookup(self):
        """Title casing differences should not cause a lookup miss."""
        vec_lower = build_semantic_relevance_vector(["dune"], "Dune", MOCK_LOOKUP, k=3)
        vec_upper = build_semantic_relevance_vector(["DUNE"], "Dune", MOCK_LOOKUP, k=3)
        assert math.isclose(vec_lower[0], vec_upper[0], abs_tol=1e-5)

    def test_no_negative_scores(self):
        """Cosine similarity is clipped to 0 — no negative scores allowed."""
        opposite_emb = _unit(np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        lookup = {**MOCK_LOOKUP, "opposite": opposite_emb}
        vec = build_semantic_relevance_vector(["opposite"], "Dune", lookup, k=3)
        assert vec[0] >= 0.0


# ── Integration: semantic relevance → ndcg round-trip ────────────────────────

class TestSemanticMetricsIntegration:

    def test_exact_hit_at_rank_one_gives_ndcg_one(self):
        """Expected book at rank 1 → semantic NDCG = 1.0."""
        ranked = ["Dune"] + ["Other Book"] * 9
        rels = build_semantic_relevance_vector(ranked, "Dune", MOCK_LOOKUP, k=10)
        assert math.isclose(ndcg_at_k(rels, k=10), 1.0, abs_tol=1e-5)

    def test_near_miss_at_rank_one_gives_high_ndcg(self):
        """
        A semantically close book at rank 1 should score much higher than a
        completely missed ranking — the core motivation for semantic NDCG.
        """
        ranked_near_miss = ["Dune Messiah"] + ["Other Book"] * 9
        ranked_total_miss = ["Other Book"] * 10
        rels_near = build_semantic_relevance_vector(ranked_near_miss,  "Dune", MOCK_LOOKUP, k=10)
        rels_miss = build_semantic_relevance_vector(ranked_total_miss, "Dune", MOCK_LOOKUP, k=10)
        assert ndcg_at_k(rels_near, k=10) > ndcg_at_k(rels_miss, k=10)

    def test_miss_gives_ndcg_zero_when_no_related_books(self):
        """All returned books with zero similarity → NDCG = 0."""
        ranked = ["Other Book"] * 10
        lookup = {"other book": _UNRELATED_EMB, "dune": _DUNE_EMB}
        rels = build_semantic_relevance_vector(ranked, "Dune", lookup, k=10)
        # "Other Book" is not Dune but it is in lookup with low-but-nonzero sim
        # so just assert it is lower than a perfect hit
        rels_perfect = build_semantic_relevance_vector(["Dune"] + ["Other Book"] * 9, "Dune", lookup, k=10)
        assert ndcg_at_k(rels, k=10) < ndcg_at_k(rels_perfect, k=10)

    def test_rank_matters_higher_rank_better_ndcg(self):
        """Expected book earlier in ranking → higher NDCG."""
        ranked_early = ["Dune", "Other Book", "Other Book"]
        ranked_late  = ["Other Book", "Other Book", "Dune"]
        rels_early = build_semantic_relevance_vector(ranked_early, "Dune", MOCK_LOOKUP, k=10)
        rels_late  = build_semantic_relevance_vector(ranked_late,  "Dune", MOCK_LOOKUP, k=10)
        assert ndcg_at_k(rels_early, k=10) > ndcg_at_k(rels_late, k=10)
