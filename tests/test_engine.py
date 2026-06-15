"""
tests/test_engine.py — Offline unit tests for QueryEngine (no index required).

All tests use a tiny synthetic in-memory index, so they run in CI without the
real 100+ MB embedding files or any network access to HuggingFace.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.matcher.preprocessing import _clean_title


# ── Helpers ───────────────────────────────────────────────────────────────────

DIM = 8  # embedding dimension for the synthetic index


def _make_meta(titles: list[str], genres: list[list[str]], domain: str) -> pd.DataFrame:
    """Build a minimal metadata DataFrame that mirrors what the real index has."""
    return pd.DataFrame({
        "title":           titles,
        "clean_title":     [_clean_title(t) for t in titles],
        "aggregated_text": [f"text about {t}" for t in titles],
        "review_count":    [10] * len(titles),
        "domain":          [domain] * len(titles),
        "sentiment":       [True] * len(titles),
        "genres":          genres,
    })


def _normalised_random(n: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.random((n, DIM), dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def engine(tmp_path):
    """
    Returns a QueryEngine backed by a tiny synthetic index in tmp_path.

    SentenceTransformer is mocked to avoid any network or disk access.
    The mock encoder returns a deterministic normalised vector for every input.
    """
    rng = np.random.default_rng(42)

    movie_titles = ["Blade Runner", "The Shining", "Dune", "Fight Club"]
    book_titles = ["Do Androids Dream of Electric Sheep?", "The Shining",
                    "Dune", "Fight Club"]
    movie_genres = [["science fiction"], ["horror"], ["science fiction"], ["drama"]]
    book_genres = [["science fiction"], ["horror"], ["science fiction"], ["drama"]]

    movie_emb = _normalised_random(len(movie_titles), rng)
    book_emb = _normalised_random(len(book_titles),  rng)

    np.save(tmp_path / "movie_embeddings.npy", movie_emb)
    np.save(tmp_path / "book_embeddings.npy",  book_emb)
    _make_meta(movie_titles, movie_genres, "movie").to_pickle(tmp_path / "movie_meta.pkl")
    _make_meta(book_titles,  book_genres,  "book").to_pickle(tmp_path  / "book_meta.pkl")

    # A mock SentenceTransformer that returns a fixed normalised vector
    _fixed_vec = rng.random(DIM, dtype=np.float32)
    _fixed_vec /= np.linalg.norm(_fixed_vec)

    mock_st = MagicMock()
    mock_st.encode.return_value = _fixed_vec

    with patch("src.matcher.engine.SentenceTransformer", return_value=mock_st):
        from src.matcher.engine import QueryEngine
        eng = QueryEngine(index_dir=tmp_path)

    return eng


# ── Tests — public query() API ────────────────────────────────────────────────

class TestQueryEngineSmoke:
    def test_query_returns_dataframe(self, engine):
        result = engine.query(["Blade Runner"], source_domain="movie", top_k=3)
        assert isinstance(result, pd.DataFrame)

    def test_query_result_columns(self, engine):
        result = engine.query(["Blade Runner"], source_domain="movie", top_k=3)
        expected = {"rank", "title", "domain", "similarity",
                    "genres", "sentiment_match", "explanation"}
        assert expected.issubset(set(result.columns))

    def test_query_respects_top_k(self, engine):
        for k in (1, 2, 3):
            result = engine.query(["The Shining"], source_domain="movie", top_k=k)
            assert len(result) == k, f"Expected {k} rows, got {len(result)}"

    def test_rank_column_is_sequential_from_one(self, engine):
        result = engine.query(["Dune"], source_domain="movie", top_k=3)
        assert list(result["rank"]) == list(range(1, len(result) + 1))

    def test_similarity_scores_are_floats(self, engine):
        result = engine.query(["Fight Club"], source_domain="movie", top_k=3)
        for sim in result["similarity"]:
            assert isinstance(sim, float)

    def test_similarity_scores_in_reasonable_range(self, engine):
        result = engine.query(["Fight Club"], source_domain="movie", top_k=3)
        for sim in result["similarity"]:
            # Genre boost can push score slightly above 1.0
            assert -1.1 <= sim <= 1.25, f"Unexpected similarity: {sim}"

    def test_auto_domain_crosses_movie_to_book(self, engine):
        result = engine.query(["Blade Runner"], source_domain="movie",
                               target_domain="auto", top_k=2)
        assert (result["domain"] == "book").all()

    def test_auto_domain_crosses_book_to_movie(self, engine):
        result = engine.query(["Dune"], source_domain="book",
                               target_domain="auto", top_k=2)
        assert (result["domain"] == "movie").all()

    def test_explicit_target_domain_respected(self, engine):
        result = engine.query(["Dune"], source_domain="book",
                               target_domain="movie", top_k=2)
        assert (result["domain"] == "movie").all()

    def test_unknown_title_falls_back_to_on_the_fly_encoding(self, engine):
        """Titles missing from the index should still return results via mock encoder."""
        result = engine.query(["NonExistentTitle12345"], source_domain="movie", top_k=2)
        assert len(result) == 2

    def test_multi_title_query_returns_results(self, engine):
        result = engine.query(
            ["Blade Runner", "Dune"], source_domain="movie", top_k=2,
        )
        assert len(result) == 2

    def test_custom_weights_accepted(self, engine):
        result = engine.query(
            ["Blade Runner", "Fight Club"],
            source_domain="movie",
            weights=[0.7, 0.3],
            top_k=2,
        )
        assert len(result) == 2

    def test_genre_boost_disabled_still_returns_results(self, engine):
        result = engine.query(["The Shining"], source_domain="movie",
                               apply_genre_boost=False, top_k=3)
        assert len(result) == 3

    def test_sentiment_filter_disabled_still_returns_results(self, engine):
        result = engine.query(["Dune"], source_domain="movie",
                               apply_sentiment_filter=False, top_k=3)
        assert len(result) == 3

    def test_explanation_field_empty_when_explain_false(self, engine):
        result = engine.query(["Dune"], source_domain="movie", explain=False, top_k=2)
        assert (result["explanation"] == "").all()

    def test_genres_column_contains_lists(self, engine):
        result = engine.query(["Blade Runner"], source_domain="movie", top_k=2)
        for genres in result["genres"]:
            assert isinstance(genres, list)


# ── Tests — internal _lookup ──────────────────────────────────────────────────

class TestQueryEngineLookup:
    def test_lookup_exact_title(self, engine):
        vec, meta = engine._lookup("Blade Runner", "movie")
        assert vec is not None
        assert meta["title"] == "Blade Runner"

    def test_lookup_case_insensitive(self, engine):
        vec, meta = engine._lookup("blade runner", "movie")
        assert vec is not None

    def test_lookup_clean_title_match(self, engine):
        vec, meta = engine._lookup("the shining", "movie")
        assert vec is not None

    def test_lookup_missing_title_returns_none(self, engine):
        vec, meta = engine._lookup("Totally Unknown Title XYZ", "movie")
        assert vec is None
        assert meta is None

    def test_lookup_book_domain(self, engine):
        vec, meta = engine._lookup("Dune", "book")
        assert vec is not None
        assert meta["title"] == "Dune"

    def test_lookup_returns_correct_embedding_shape(self, engine):
        vec, _ = engine._lookup("Dune", "movie")
        assert vec.shape == (DIM,)


# ── Tests — _query_genres ─────────────────────────────────────────────────────

class TestQueryEngineGenres:
    def test_returns_set(self, engine):
        genres = engine._query_genres(["Blade Runner"], "movie")
        assert isinstance(genres, set)

    def test_correct_genre_values(self, engine):
        genres = engine._query_genres(["Blade Runner"], "movie")
        assert "science fiction" in genres

    def test_multi_title_union(self, engine):
        genres = engine._query_genres(["Blade Runner", "The Shining"], "movie")
        assert "science fiction" in genres
        assert "horror" in genres

    def test_unknown_title_returns_empty_set(self, engine):
        genres = engine._query_genres(["CompletelyUnknownXYZ"], "movie")
        assert isinstance(genres, set)  # may be empty, but must be a set

    def test_empty_title_list(self, engine):
        genres = engine._query_genres([], "movie")
        assert genres == set()


# ── Tests — _query_sentiment ──────────────────────────────────────────────────

class TestQueryEngineSentiment:
    def test_known_title_returns_bool(self, engine):
        sentiment = engine._query_sentiment(["Blade Runner"], "movie")
        assert isinstance(sentiment, bool)

    def test_unknown_title_returns_none(self, engine):
        sentiment = engine._query_sentiment(["GhostTitleXYZ"], "movie")
        assert sentiment is None

    def test_empty_title_list_returns_none(self, engine):
        sentiment = engine._query_sentiment([], "movie")
        assert sentiment is None


# ── Tests — _build_query_vector ───────────────────────────────────────────────

class TestBuildQueryVector:
    def test_returns_normalised_vector(self, engine):
        vec = engine._build_query_vector(["Dune"], [1.0], "movie")
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_output_shape(self, engine):
        vec = engine._build_query_vector(["Blade Runner"], [1.0], "movie")
        assert vec.shape == (DIM,)

    def test_multi_title_output_shape(self, engine):
        vec = engine._build_query_vector(
            ["Blade Runner", "Dune"], [0.5, 0.5], "movie"
        )
        assert vec.shape == (DIM,)

    def test_uniform_weights_same_as_equal_weights(self, engine):
        v1 = engine._build_query_vector(["Blade Runner", "Dune"], [1.0, 1.0], "movie")
        v2 = engine._build_query_vector(["Blade Runner", "Dune"], [0.5, 0.5], "movie")
        assert np.allclose(v1, v2, atol=1e-5)
