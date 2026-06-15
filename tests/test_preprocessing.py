"""
tests/test_preprocessing.py — Unit tests for all preprocessing helpers.

No index, no model, no network required — pure string/data manipulation only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.matcher.preprocessing import _parse_genres, _clean_title, _agg_movies, _agg_books


# ── _parse_genres ─────────────────────────────────────────────────────────────

class TestParseGenres:
    def test_none_returns_empty(self):
        assert _parse_genres(None) == []

    def test_nan_float_returns_empty(self):
        assert _parse_genres(float("nan")) == []

    def test_empty_string_returns_empty(self):
        assert _parse_genres("") == []

    def test_string_nan_returns_empty(self):
        assert _parse_genres("nan") == []

    def test_string_none_returns_empty(self):
        assert _parse_genres("none") == []

    def test_comma_separated_string(self):
        result = _parse_genres("Sci-Fi, Action, Drama")
        assert result == ["sci-fi", "action", "drama"]

    def test_pipe_separated_string(self):
        result = _parse_genres("Sci-Fi|Action|Drama")
        assert result == ["sci-fi", "action", "drama"]

    def test_semicolon_separated_string(self):
        result = _parse_genres("Sci-Fi;Action")
        assert result == ["sci-fi", "action"]

    def test_single_genre_string(self):
        result = _parse_genres("Horror")
        assert result == ["horror"]

    def test_list_input_lowercased(self):
        result = _parse_genres(["Sci-Fi", "Action"])
        assert result == ["sci-fi", "action"]

    def test_list_with_empty_strings_filtered(self):
        result = _parse_genres(["Sci-Fi", "", "  "])
        # Empty and whitespace-only entries should be excluded
        assert "sci-fi" in result
        assert "" not in result

    def test_extra_whitespace_stripped(self):
        result = _parse_genres("  Horror ,  Drama  ")
        assert result == ["horror", "drama"]

    def test_returns_list(self):
        assert isinstance(_parse_genres("Horror"), list)

    def test_numpy_nan_returns_empty(self):
        assert _parse_genres(np.nan) == []


# ── _clean_title ──────────────────────────────────────────────────────────────

class TestCleanTitle:
    def test_lowercased(self):
        assert _clean_title("The SHINING") == "the shining"

    def test_punctuation_removed(self):
        assert _clean_title("Catch-22!") == "catch22"

    def test_leading_trailing_whitespace_stripped(self):
        assert _clean_title("  Dune  ") == "dune"

    def test_multiple_internal_spaces_collapsed(self):
        assert _clean_title("Fight   Club") == "fight club"

    def test_apostrophe_removed(self):
        result = _clean_title("Schindler's List")
        assert "'" not in result
        assert "schindlers list" == result

    def test_none_coerced_to_string(self):
        result = _clean_title(None)
        assert isinstance(result, str)

    def test_empty_string(self):
        assert _clean_title("") == ""

    def test_returns_string(self):
        assert isinstance(_clean_title("Blade Runner"), str)

    def test_unicode_title(self):
        # Should not raise; just lower + strip punctuation
        result = _clean_title("Léon: The Professional")
        assert isinstance(result, str)

    def test_numbers_preserved(self):
        result = _clean_title("2001: A Space Odyssey")
        assert "2001" in result


# ── _agg_movies ───────────────────────────────────────────────────────────────

class TestAggMovies:
    def _row(self, **kwargs) -> pd.Series:
        defaults = {
            "plot_summary":   "",
            "review_detail":  "",
            "review_summary": "",
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_returns_string(self):
        assert isinstance(_agg_movies(self._row()), str)

    def test_uses_plot_summary(self):
        row = self._row(plot_summary="A great plot.")
        assert "A great plot." in _agg_movies(row)

    def test_uses_review_detail_when_no_plot(self):
        row = self._row(review_detail="Fantastic movie.")
        assert "Fantastic movie." in _agg_movies(row)

    def test_uses_review_summary_as_fallback(self):
        row = self._row(review_summary="Brilliant.")
        assert "Brilliant." in _agg_movies(row)

    def test_combines_multiple_fields(self):
        row = self._row(plot_summary="Plot.", review_detail="Review.")
        result = _agg_movies(row)
        assert "Plot." in result
        assert "Review." in result

    def test_skips_nan_string(self):
        row = self._row(plot_summary="nan", review_detail="Good film.")
        result = _agg_movies(row)
        assert "nan" not in result
        assert "Good film." in result

    def test_skips_none_string(self):
        row = self._row(plot_summary="none", review_detail="Wonderful.")
        assert "none" not in _agg_movies(row)

    def test_max_length_2000(self):
        long_text = "x" * 3000
        row = self._row(plot_summary=long_text)
        assert len(_agg_movies(row)) <= 2000

    def test_empty_row_returns_empty_string(self):
        assert _agg_movies(self._row()) == ""


# ── _agg_books ────────────────────────────────────────────────────────────────

class TestAggBooks:
    def _row(self, **kwargs) -> pd.Series:
        defaults = {
            "plot_summary":  "",
            "review_detail": "",
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_returns_string(self):
        assert isinstance(_agg_books(self._row()), str)

    def test_uses_plot_summary(self):
        row = self._row(plot_summary="Epic fantasy saga.")
        assert "Epic fantasy saga." in _agg_books(row)

    def test_uses_review_detail_when_no_plot(self):
        row = self._row(review_detail="A must-read.")
        assert "A must-read." in _agg_books(row)

    def test_combines_fields(self):
        row = self._row(plot_summary="The plot.", review_detail="The review.")
        result = _agg_books(row)
        assert "The plot." in result
        assert "The review." in result

    def test_skips_nan_string(self):
        row = self._row(plot_summary="nan", review_detail="Compelling.")
        result = _agg_books(row)
        assert "nan" not in result

    def test_max_length_2000(self):
        long_text = "y" * 3000
        row = self._row(plot_summary=long_text)
        assert len(_agg_books(row)) <= 2000

    def test_empty_row_returns_empty_string(self):
        assert _agg_books(self._row()) == ""
