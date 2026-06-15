from __future__ import annotations

import ast
import html
import re

import numpy as np
import pandas as pd


# ── Private helpers ───────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BRACKET_QUOTE_RE = re.compile(r"^[\[\]'\"{}\s]+|[\[\]'\"{}\s]+$")


def _clean_genre_token(token) -> str:
    """Strip HTML tags/entities and stray bracket/quote chars; lowercase."""
    text = str(token)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _BRACKET_QUOTE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _flatten_genre_value(value) -> list[str]:
    """Flatten a list/dict/list-of-dicts genre value into name strings."""
    out: list[str] = []
    if isinstance(value, dict):
        name = value.get("name") or value.get("genre") or value.get("title")
        if name:
            out.append(str(name))
        return out
    if isinstance(value, (list, tuple, set)):
        for item in value:
            out.extend(_flatten_genre_value(item))
        return out
    return [str(value)]


def _parse_genres(raw_value) -> list[str]:
    """
    Converts a raw genre cell value into a normalised list of strings.

    Handles:
      - NaN / None                      -> []
      - Comma-separated str             -> ['sci-fi', 'action']
      - Pipe-separated str              -> ['sci-fi', 'action']
      - Already a list                  -> lowercased copy
      - Stringified list / list-of-dicts
        (e.g. TMDB's
        "[{'id': 28, 'name': 'Action'}]") -> ['action']
      - Genre strings containing stray
        HTML markup / entities          -> tags stripped, entities decoded
    """
    if raw_value is None:
        return []
    if isinstance(raw_value, float) and np.isnan(raw_value):
        return []

    if isinstance(raw_value, (list, tuple, set, dict)):
        tokens = _flatten_genre_value(raw_value)
        cleaned = [_clean_genre_token(g) for g in tokens]
        return [g for g in cleaned if g]

    raw_str = str(raw_value).strip()
    if not raw_str or raw_str.lower() in ("nan", "none", ""):
        return []

    # Stringified list / list-of-dicts, e.g. "['Action', 'Adventure']" or
    # "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"
    if raw_str.startswith("[") and raw_str.endswith("]"):
        try:
            parsed = ast.literal_eval(raw_str)
        except (ValueError, SyntaxError):
            parsed = None
        if parsed is not None:
            tokens = _flatten_genre_value(parsed)
            cleaned = [_clean_genre_token(g) for g in tokens]
            return [g for g in cleaned if g]

    # Strip HTML tags and decode entities (e.g. "&amp;" -> "&") on the whole
    # string FIRST, so that ';'-terminated entities aren't mangled by the
    # ';' -> ',' separator normalisation below.
    raw_str = html.unescape(_HTML_TAG_RE.sub(" ", raw_str))

    normalised = raw_str.lower().replace("|", ",").replace(";", ",")
    cleaned = [_clean_genre_token(g) for g in normalised.split(",")]
    return [g for g in cleaned if g]


def _clean_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ",
                  re.sub(r"[^\w\s]", "", str(title).lower().strip())).strip()


def _agg_movies(row: pd.Series) -> str:
    """
    Builds an embedding-ready text string for a movie row.
    Priority: plot_summary → review_detail → review_summary.
    """
    parts = []
    for col in ("plot_summary", "review_detail", "review_summary"):
        val = str(row.get(col) or "").strip()
        if val and val.lower() not in ("nan", "none", ""):
            parts.append(val)
    return " ".join(parts)[:2000]


def _agg_books(row: pd.Series) -> str:
    """
    Builds an embedding-ready text string for a book row.
    Priority: plot_summary → review_detail.
    """
    parts = []
    for col in ("plot_summary", "review_detail"):
        val = str(row.get(col) or "").strip()
        if val and val.lower() not in ("nan", "none", ""):
            parts.append(val)
    return " ".join(parts)[:2000]

