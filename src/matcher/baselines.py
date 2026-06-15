from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from src.matcher.config import INDEX_DIR

warnings.filterwarnings("ignore")


# ── Baselines ─────────────────────────────────────────────────────────────────

class PopularityBaseline:
    """
    The dumbest possible recommender.

    Ignores the query entirely and always returns the most-reviewed items.
    Sets the performance floor: if the semantic model cannot beat this,
    something is wrong.  On the adaptation-pair test set, which includes
    niche titles, this baseline should fail badly.
    """

    name = "Popularity Baseline"

    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        self.movie_meta = pd.read_pickle(index_dir / "movie_meta.pkl")
        self.book_meta = pd.read_pickle(index_dir / "book_meta.pkl")

        # Pre-sort once so every query is O(1)
        self._top_books = (self.book_meta
                            .sort_values("review_count", ascending=False)
                            .reset_index(drop=True))
        self._top_movies = (self.movie_meta
                            .sort_values("review_count", ascending=False)
                            .reset_index(drop=True))

    def query(
        self,
        titles:        list[str],
        source_domain: str,
        target_domain: str = "auto",
        top_k:         int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        if target_domain == "auto":
            target_domain = "book" if source_domain == "movie" else "movie"

        pool = self._top_books if target_domain == "book" else self._top_movies
        top = pool.head(top_k)

        return pd.DataFrame([
            {
                "rank":        rank + 1,
                "title":       row["title"],
                "domain":      target_domain,
                "similarity":  float(len(pool) - rank) / len(pool),  # proxy score
                "explanation": f"Ranked #{rank + 1} most-reviewed {target_domain}",
            }
            for rank, (_, row) in enumerate(top.iterrows())
        ])


class TFIDFBaseline:
    """
    TF-IDF bag-of-words baseline.

    Uses the same aggregated-text pipeline as QueryEngine (plot + reviews,
    cosine similarity, weighted composite queries) but replaces the dense
    sentence-transformer embedding with a sparse TF-IDF vector.

    This isolates the value added by the sentence-transformer model: if the
    embedding model beats TF-IDF by a meaningful margin, the semantic
    representations are doing something beyond simple keyword matching.
    """

    name = "TF-IDF Baseline"

    def __init__(
        self,
        index_dir:    Path = INDEX_DIR,
        max_features: int  = 40_000,
    ) -> None:
        print(f"[{self.name}] Building TF-IDF index …")
        self.movie_meta = pd.read_pickle(index_dir / "movie_meta.pkl")
        self.book_meta  = pd.read_pickle(index_dir / "book_meta.pkl")

        movie_texts = self.movie_meta["aggregated_text"].fillna("").tolist()
        book_texts  = self.book_meta["aggregated_text"].fillna("").tolist()

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,       # log(1 + tf) — dampens extreme counts
            strip_accents="unicode",
        )
        all_vecs = self.vectorizer.fit_transform(movie_texts + book_texts)
        all_vecs = sk_normalize(all_vecs, norm="l2")

        n_movies         = len(self.movie_meta)
        self.movie_vecs  = all_vecs[:n_movies]
        self.book_vecs   = all_vecs[n_movies:]
        self._vocab_size = len(self.vectorizer.vocabulary_)

        print(
            f"  ✓ TF-IDF index ready.  Vocab: {self._vocab_size:,} terms, "
            f"{n_movies:,} movies, {len(self.book_meta):,} books"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_vec(self, title: str, domain: str):
        """
        Returns the L2-normalised TF-IDF vector for the best-matching title,
        or ``None`` when the title is not in the index.

        Lookup order mirrors QueryEngine._lookup:
          1. Exact title match (case-insensitive)
          2. Exact clean_title match
          3. Substring match on clean_title
        """
        meta = self.movie_meta if domain == "movie" else self.book_meta
        vecs = self.movie_vecs if domain == "movie" else self.book_vecs

        mask = meta["title"].str.lower() == title.lower()
        if not mask.any():
            mask = meta["clean_title"].str.lower() == title.lower()
        if not mask.any():
            mask = meta["clean_title"].str.contains(
                title.lower().strip(), regex=False, na=False
            )
        if not mask.any():
            return None

        idx = meta.loc[mask, "review_count"].idxmax()
        return vecs[idx]

    # ── Public interface — mirrors QueryEngine.query ──────────────────────────

    def query(
        self,
        titles:        list[str],
        source_domain: str,
        target_domain: str = "auto",
        weights:       list[float] | None = None,
        top_k:         int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        if target_domain == "auto":
            target_domain = "book" if source_domain == "movie" else "movie"
        if weights is None:
            weights = [1.0 / len(titles)] * len(titles)

        # Weighted composite query vector
        composite = None
        total_w = 0.0
        for title, weight in zip(titles, weights):
            vec = self._get_vec(title, source_domain)
            if vec is None:
                # Encode on-the-fly via the fitted vectorizer
                vec = sk_normalize(self.vectorizer.transform([title]), norm="l2")
            composite = vec * weight if composite is None else composite + vec * weight
            total_w += weight

        if composite is None or total_w == 0:
            return pd.DataFrame()

        composite = sk_normalize(composite, norm="l2")

        cand_vecs = self.book_vecs if target_domain == "book" else self.movie_vecs
        cand_meta = self.book_meta if target_domain == "book" else self.movie_meta

        sims = (
            (cand_vecs @ composite.T).toarray().flatten()
            if issparse(composite)
            else (cand_vecs @ composite.flatten())
        )

        top_idx = np.argsort(sims)[::-1][:top_k]

        return pd.DataFrame([
            {
                "rank":        rank + 1,
                "title":       cand_meta.iloc[idx]["title"],
                "domain":      target_domain,
                "similarity":  round(float(sims[idx]), 4),
                "explanation": "TF-IDF bag-of-words match",
            }
            for rank, idx in enumerate(top_idx)
        ])
