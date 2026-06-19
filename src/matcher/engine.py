from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.matcher.config import INDEX_DIR, MODEL_NAME, GEMINI_API_KEY, GEMINI_MODEL_NAME
from src.matcher.preprocessing import _clean_title


class QueryEngine:
    """
    Cross-domain recommendation engine with three-stage hybrid re-ranking.

    Stage 1 — Dense Semantic Retrieval
        Embeds the query and retrieves the top ``semantic_pool_size`` most
        atmospherically similar candidates via cosine similarity.

    Stage 2 — Genre Jaccard Boost
        Re-scores those candidates using genre overlap (Jaccard similarity)
        to surface items that also share the same taxonomic context.

    Stage 3 — Mood Jaccard Boost
        Further re-scores using mood/atmosphere tag overlap so that books
        and movies that feel emotionally similar rise to the top.  Mood
        tags are assigned offline by Gemini during index building and
        stored in the ``mood_tags`` metadata column.

    Parameters
    ----------
    index_dir : Path
        Folder that holds ``*_embeddings.npy`` and ``*_meta.pkl`` files.
        Defaults to the module-level ``INDEX_DIR``.
    """

    # Human-readable name used by the Evaluator
    name = "VibeMatcher (Semantic + Genre + Mood)"

    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        print("Loading index …")
        self.movie_emb = np.load(index_dir / "movie_embeddings.npy")
        self.book_emb = np.load(index_dir / "book_embeddings.npy")
        self.movie_meta = pd.read_pickle(index_dir / "movie_meta.pkl")
        self.book_meta = pd.read_pickle(index_dir / "book_meta.pkl")
        self.embed_model = SentenceTransformer(MODEL_NAME)
        self._dim = self.movie_emb.shape[1]

        # Lazily initialised — only created when explain=True is requested
        self._llm = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_llm(self):
        """Lazily initialise the Gemini client on first use (google-genai SDK)."""
        if self._llm is None:
            try:
                from google import genai  # pip install google-genai
            except ImportError as exc:
                raise ImportError(
                    "google-genai is required for explanations. "
                    "Install it with:  pip install google-genai"
                ) from exc

            if not GEMINI_API_KEY:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Set it in your .env file "
                    "locally or as a Secret in your HuggingFace Space settings."
                )

            self._llm = genai.Client(api_key=GEMINI_API_KEY)
        return self._llm

    def _lookup(self, title: str, domain: str) -> tuple[np.ndarray | None, pd.Series | None]:
        """
        Returns ``(embedding_vector, metadata_row)`` for the best-matching title.

        Lookup priority:
          1. Case-insensitive exact match on the ``title`` column.
          2. Exact match on ``clean_title``.
          3. Substring match on ``clean_title`` (most-reviewed title wins).

        Returns ``(None, None)`` when no match is found.
        """
        meta = self.movie_meta if domain == "movie" else self.book_meta
        emb  = self.movie_emb  if domain == "movie" else self.book_emb

        # Pass 1 — exact title (case-insensitive)
        mask = meta["title"].str.lower() == title.lower()
        if mask.sum() == 1:
            idx = mask.idxmax()
            return emb[idx], meta.loc[idx]
        if mask.sum() > 1:
            idx = meta.loc[mask, "review_count"].idxmax()
            return emb[idx], meta.loc[idx]

        # Pass 2 — exact clean_title
        ctitle = _clean_title(title)
        mask2 = meta["clean_title"] == ctitle
        if mask2.any():
            idx = meta.loc[mask2, "review_count"].idxmax()
            return emb[idx], meta.loc[idx]

        # Pass 3 — substring on clean_title
        mask3 = meta["clean_title"].str.contains(ctitle, regex=False, na=False)
        if mask3.any():
            idx = meta.loc[mask3, "review_count"].idxmax()
            return emb[idx], meta.loc[idx]

        return None, None

    def _build_query_vector(
        self,
        titles: list[str],
        weights: list[float],
        source_domain: str,
    ) -> np.ndarray:
        """Weighted average of source title embeddings, L2-normalised."""
        total = sum(weights)
        weights = [w / total for w in weights]
        composite = np.zeros(self._dim, dtype=np.float32)

        for title, weight in zip(titles, weights):
            vec, _ = self._lookup(title, source_domain)
            if vec is None:
                print(f"  '{title}' not in index — encoding on-the-fly.")
                vec = self.embed_model.encode(
                    title,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).astype(np.float32)
            composite += weight * vec

        norm = np.linalg.norm(composite)
        return composite / norm if norm > 0 else composite

    def _query_genres(self, titles: list[str], source_domain: str) -> set[str]:
        """Returns the union of all genre tags from the given source titles."""
        query_genres: set[str] = set()
        for title in titles:
            _, meta_row = self._lookup(title, source_domain)
            if meta_row is not None:
                genres = meta_row.get("genres", [])
                if isinstance(genres, list):
                    query_genres.update(genres)
        return query_genres

    def _query_moods(self, titles: list[str], source_domain: str) -> set[str]:
        """
        Returns the union of all mood tags from the given source titles.

        Mirrors ``_query_genres`` exactly — gracefully returns an empty set
        when ``mood_tags`` is absent from the index (e.g. the index was built
        without ``use_gemini_moods=True``).
        """
        query_moods: set[str] = set()
        for title in titles:
            _, meta_row = self._lookup(title, source_domain)
            if meta_row is not None:
                moods = meta_row.get("mood_tags", [])
                if isinstance(moods, list):
                    query_moods.update(moods)
        return query_moods

    def _query_sentiment(self, titles: list[str], source_domain: str) -> bool | None:
        """
        Returns the majority sentiment polarity of the source titles,
        or ``None`` if none of the titles are found in the index.
        """
        meta = self.movie_meta if source_domain == "movie" else self.book_meta
        labels: list[bool] = []

        for title in titles:
            mask = meta["title"].str.lower() == title.lower()
            if not mask.any():
                mask = meta["clean_title"] == _clean_title(title)
            if mask.any():
                idx = meta.loc[mask, "review_count"].idxmax()
                labels.append(bool(meta.loc[idx, "sentiment"]))

        if not labels:
            return None
        return sum(labels) >= len(labels) / 2

    def _explain_batch(
        self,
        source_titles: list[str],
        source_domain: str,
        candidates:    list[dict],  # list of {"title": str, "domain": str, "text": str}
    ) -> dict[str, str]:
        """
        Single Gemini call for ALL recommendations at once.
        Returns a dict mapping rec_title -> one-sentence explanation.
        Retries up to 3 times with exponential back-off on failure.
        """
        import json

        entries = "\n".join(
            f"{i+1}. \"{c['title']}\" ({c['domain']}): {c['text'][:300]}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            f"A user likes the following {source_domain}(s): {', '.join(source_titles)}.\n\n"
            f"The system recommends these titles:\n{entries}\n\n"
            f"For each recommended title, write exactly one sentence (max 30 words) explaining "
            f"why a fan of {', '.join(source_titles)} would enjoy it. "
            f"Be specific about shared themes, tone, or atmosphere.\n\n"
            f"Respond ONLY with a valid JSON object where each key is the exact title string "
            f"and the value is the one-sentence explanation. No markdown, no preamble."
        )
        llm = self._get_llm()
        from google.genai import types as _genai_types

        config = _genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=2048,
        )

        for attempt in range(3):
            try:
                response = llm.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=prompt,
                    config=config,
                )
                raw = (response.text or "").strip()
                # Belt-and-braces: still strip stray markdown fences if the
                # model ever ignores response_mime_type.
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                if not raw:
                    raise ValueError("Empty response body from Gemini")
                return json.loads(raw)
            except Exception as exc:
                wait = 30 * (attempt + 1)
                print(f"  Gemini batch error (attempt {attempt + 1}/3): {exc} — retrying in {wait}s")
                if attempt < 2:
                    time.sleep(wait)
        # Fallback: surface the failure instead of hiding it
        print(
            f"  [explain] Gave up after 3 attempts for {len(candidates)} candidates — "
            "explanations will show as unavailable in the UI."
        )
        return {c["title"]: "(explanation unavailable)" for c in candidates}

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        titles:                  list[str],
        source_domain:           str,
        target_domain:           str = "auto",
        weights:                 list[float] | None = None,
        top_k:                   int = 5,
        apply_sentiment_filter:  bool = True,
        apply_genre_boost:       bool = True,
        genre_boost_weight:      float = 0.15,
        apply_mood_boost:        bool = True,
        mood_boost_weight:       float = 0.20,
        semantic_pool_size:      int = 100,
        explain:                 bool = False,
    ) -> pd.DataFrame:
        """
        Retrieve cross-domain recommendations for one or more source titles.

        Parameters
        ----------
        titles:
            Source title(s) the user likes.
        source_domain:
            ``"movie"`` or ``"book"``.
        target_domain:
            ``"movie"``, ``"book"``, or ``"auto"`` (crosses domain automatically).
        weights:
            Optional blend weights, same length as *titles*.  Uniform by default.
        top_k:
            Number of final recommendations returned.
        apply_sentiment_filter:
            Soft-penalise (×0.8) candidates whose sentiment polarity differs from
            the query.
        apply_genre_boost:
            Enable Stage-2 Jaccard genre re-ranking.
        genre_boost_weight:
            Maximum score bonus added by a perfect genre overlap (default 0.15).
        apply_mood_boost:
            Enable Stage-3 Jaccard mood re-ranking.  Silently skipped when the
            index does not contain ``mood_tags`` (built without Gemini moods).
        mood_boost_weight:
            Maximum score bonus added by a perfect mood overlap (default 0.20).
        semantic_pool_size:
            Size of the Stage-1 candidate pool before re-ranking stages.
        explain:
            Generate a one-sentence Gemini explanation per result.
            Requires ``GEMINI_API_KEY`` and ``google-generativeai`` installed.

        Returns
        -------
        pd.DataFrame
            Columns: ``rank``, ``title``, ``domain``, ``similarity``,
            ``genres``, ``mood_tags``, ``sentiment_match``, ``explanation``.
        """
        if weights is None:
            weights = [1.0] * len(titles)

        if target_domain == "auto":
            target_domain = "book" if source_domain == "movie" else "movie"

        # ── Stage 1: Dense semantic retrieval ─────────────────────────────────
        query_vec = self._build_query_vector(titles, weights, source_domain)

        cand_meta = (
            self.book_meta if target_domain == "book" else self.movie_meta
        ).copy()
        cand_emb = self.book_emb if target_domain == "book" else self.movie_emb

        # Cosine similarity — vectors are already L2-normalised so dot product suffices
        sims = (cand_emb @ query_vec).astype(float)

        # Soft sentiment penalty
        query_pol = self._query_sentiment(titles, source_domain)
        if apply_sentiment_filter and query_pol is not None:
            mismatch = cand_meta["sentiment"].values != query_pol
            sims[mismatch] *= 0.8

        cand_meta = cand_meta.reset_index(drop=True)
        cand_meta["score"] = sims

        # ── Stage 2: Genre Jaccard boost ──────────────────────────────────────
        if apply_genre_boost:
            query_genres = self._query_genres(titles, source_domain)

            if query_genres:
                pool_size = min(semantic_pool_size, len(cand_meta))
                pool_idx = cand_meta["score"].nlargest(pool_size).index

                for idx in pool_idx:
                    raw = cand_meta.at[idx, "genres"]
                    cand_genres = set(raw) if isinstance(raw, list) else set()
                    if cand_genres:
                        intersection = query_genres & cand_genres
                        union = query_genres | cand_genres
                        jaccard = len(intersection) / len(union)
                        cand_meta.at[idx, "score"] += jaccard * genre_boost_weight
            else:
                print("  [genre boost] No genres found for query titles — skipping.")

        # ── Stage 3: Mood Jaccard boost ───────────────────────────────────────
        if apply_mood_boost:
            query_moods = self._query_moods(titles, source_domain)

            if query_moods:
                pool_size = min(semantic_pool_size, len(cand_meta))
                pool_idx = cand_meta["score"].nlargest(pool_size).index

                for idx in pool_idx:
                    raw = cand_meta.at[idx, "mood_tags"]
                    cand_moods = set(raw) if isinstance(raw, list) else set()
                    if cand_moods:
                        intersection = query_moods & cand_moods
                        union = query_moods | cand_moods
                        jaccard = len(intersection) / len(union)
                        cand_meta.at[idx, "score"] += jaccard * mood_boost_weight
            else:
                print("  [mood boost] No mood tags found for query titles — skipping.")

        # ── Final ranking ─────────────────────────────────────────────────────
        final_top = cand_meta.sort_values("score", ascending=False).head(top_k)

        result_rows: list[dict] = []
        # ── Batch explanations: 1 Gemini call for all results ────────────────
        explanations: dict[str, str] = {}
        if explain:
            candidates = [
                {
                    "title":  row["title"],
                    "domain": target_domain,
                    "text":   row.get("aggregated_text", ""),
                }
                for _, row in final_top.iterrows()
            ]
            explanations = self._explain_batch(
                source_titles=titles,
                source_domain=source_domain,
                candidates=candidates,
            )

        for rank, (_, row) in enumerate(final_top.iterrows(), start=1):
            sim = round(float(row["score"]), 4)
            sent_ok = (bool(row["sentiment"]) == query_pol if query_pol is not None else None)

            # Ensure genres/moods are always a clean list of plain strings.
            # Guard against index versions that stored pre-rendered HTML fragments.
            raw_genres = row.get("genres", [])
            genres_clean = (
                [str(g) for g in raw_genres if isinstance(g, str) and "<" not in g and g.strip()]
                if isinstance(raw_genres, (list, tuple))
                else []
            )

            raw_moods = row.get("mood_tags", [])
            moods_clean = (
                [str(m) for m in raw_moods if isinstance(m, str) and "<" not in m and m.strip()]
                if isinstance(raw_moods, (list, tuple))
                else []
            )

            result_rows.append({
                "rank":            rank,
                "title":           row["title"],
                "domain":          target_domain,
                "similarity":      sim,
                "genres":          genres_clean,
                "mood_tags":       moods_clean,
                "sentiment_match": sent_ok,
                "explanation":     explanations.get(row["title"], ""),
            })

        return pd.DataFrame(result_rows)
