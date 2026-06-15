from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    hit_rate_at_k,
    reciprocal_rank,
    build_hybrid_relevance_vector,
    build_ideal_relevance_vector,
    ndcg_at_k,
)
from src.matcher.config import INDEX_DIR, RESULTS_DIR

warnings.filterwarnings("ignore")


# ── Golden test pairs ─────────────────────────────────────────────────────────

GOLDEN_PAIRS: list[tuple[str, tuple[str, str, str]]] = [
    # (movie_title_in_index, (rank_1_book, rank_2_book, rank_3_book))
    ("The Shining",   ("Gerald's Game: A Novel", "Doctor Sleep", "Forsaken: A Novel of Art, Evil and Insanity")),
    ("Fight Club",   ("Fight Club", "Deepest Doors", "Choke")),
    ("No Country for Old Men",   ("No Country for Old Men", "Let the Good Prevail", "The Road")),
    ("The Godfather",   ("The Godfather Returns: A New Novel Based on the Corleone Family Characters Created by Mario Puzo", "The Sicilian", "The Family Corleone")),
    ("Jurassic Park",   ("Jurassic Park", "Anonymous Rex (Anonymous Rex, #1)", "The Great Zoo of China")),
    ("One Flew Over the Cuckoo's Nest",   ("The Madman's Tale", "Project 17", "The Interview Room (Paul Lucas, #1)")),
    ("Pulp Fiction",   ("Pulp Fiction", "Pulp Ink", "Loser's Town")),
    ("American Psycho",   ("AmerikÄÅ†u psihs", "Since the Layoffs", "New York Dead (Stone Barrington, #1)")),
    ("Gone Girl",   ("Gone Girl", "Or She Dies", "Only the Truth: A Novel")),
    ("The Girl with the Dragon Tattoo",   ("The Girl with the Dragon Tattoo (Millennium, #1)", "The Girl with the Dragon Tattoo - Millennium", "De vackraste")),
    ("Catch-22",   ("Catch-22", "The Master Sniper", "Flight of Eagles (Dougal Munro and Jack Carter, #3)")),
    ("The Handmaid's Tale",   ("The Handmaid's Tale", "Wither (The Chemical Garden, #1)", "Rite of Rejection (Acceptance #1)")),
    ("A Clockwork Orange",   ("A Clockwork Orange", "The Administration (The Administration, #1-7)", "Disco Bloodbath")),
    ("Dune",   ("Duin (Duin, #1)", "Arrakis - Ã¶kenplaneten", "Dune (Dune Chronicles, #1)")),
    ("Interview with the Vampire",   ("Interview with the Vampire (The Vampire Chronicles, #1)", "The Vampire Lestat", "Into the Dark")),
    ("The Silence of the Lambs",   ("The Silence of the Lambs (Hannibal Lecter, #2)", "Speak of the Devil: a Novel", "Hannibal (Hannibal Lecter, #3")),
    ("Blade Runner",   ("Do Androids Dream of Electric Sheep?", "The Human Forged", "A Scanner Darkly")),
    ("Schindler's List",   ("Schindler's Ark", "The German Doctor", "The Lady from Zagreb (Bernard Gunther, #10)")),
    ("To Kill a Mockingbird",   ("To Kill a Mockingbird", "Defending Jacob", "The Outsiders")),
    ("The Green Mile",   ("You Got Nothing Coming: Notes from a Prison Fish", "The Shawshank Redemption", "Vicious Circle (Felix Castor, #2)")),
    ("Trainspotting",   ("Get Clean", "Trainspotting", "Disco Bloodbath")),
    ("The Big Short",   ("In Bed with Wall Street: The Conspiracy Crippling Our Global Economy", "The Vulture Fund", "Liar's Poker")),
    ("Breakfast at Tiffany's",   ("Breakfast at Tiffany's", "Glitter Baby (Wynette, Texas #3)", "Beverly Hills")),
    ("Misery",   ("Sie - Misery", "Portrait of the Psychopath as a Young Woman", "Blinded")),
    ("Carrie",   ("Carrie", "Gerald's Game: A Novel", "Rabbit in Red")),
]


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs any number of models over the golden test set and produces a
    side-by-side comparison table using *semantic* NDCG.

    Semantic NDCG assigns continuous relevance scores based on embedding
    cosine similarity between each retrieved book and the expected book,
    rather than a binary 0/1 title-match.  This means semantically close
    books (same atmosphere, themes, genre) receive partial credit, making
    the metric meaningful for a vibe-matching system where finding the
    exact canonical adaptation is a harsh and misleading target.

    Hit Rate and MRR remain exact-match — they test whether the system
    can surface the specific adaptation when it exists in the corpus.

    Parameters
    ----------
    golden_pairs:
        List of ``(query_title, expected_title)`` tuples.
        Defaults to the module-level ``GOLDEN_PAIRS``.
    k:
        Metric cutoff rank (default 10).
    index_dir:
        Path to the folder holding ``book_embeddings.npy`` and
        ``book_meta.pkl``.  Defaults to the project-level ``INDEX_DIR``.
        Embeddings are loaded once at construction time.
    """

    def __init__(
        self,
        golden_pairs: list[tuple[str, str]] | None = None,
        k:            int = 10,
        index_dir:    Path | None = None,
        strictness: float = 2.0
    ) -> None:
        self.golden_pairs = golden_pairs or GOLDEN_PAIRS
        self.k = k
        self.strictness = strictness

        # ── Load book embeddings for semantic relevance ───────────────────────
        _dir = Path(index_dir) if index_dir is not None else INDEX_DIR
        self._title_to_emb: dict[str, np.ndarray] = {}

        try:
            book_emb = np.load(_dir / "book_embeddings.npy")
            book_meta = pd.read_pickle(_dir / "book_meta.pkl")
            self._title_to_emb = {
                str(title).lower().strip(): book_emb[i]
                for i, title in enumerate(book_meta["title"])
            }
            print(
                f"[Evaluator] Loaded {len(self._title_to_emb):,} book embeddings "
                f"for semantic NDCG."
            )
        except FileNotFoundError:
            print(
                "[Evaluator] WARNING: book_embeddings.npy / book_meta.pkl not found "
                f"in '{_dir}'.  Semantic NDCG will degrade to exact-match fallback "
                "(score = 1 only for the expected title, 0 otherwise)."
            )

    # ── Per-model evaluation ──────────────────────────────────────────────────

    def evaluate_model(
        self,
        model,
        source_domain: str = "movie",
        target_domain: str = "book",
    ) -> pd.DataFrame:
        """
        Evaluates *model* over every golden pair.

        Returns a DataFrame with one row per pair containing:
        ``model``, ``query_title``, ``expected_title``,
        ``hit_at_k``, ``rr``, ``semantic_ndcg_at_k``,
        ``returned_titles``, ``found_at_rank``.
        """
        model_name = getattr(model, "name", type(model).__name__)
        print(f"\n[Evaluating] {model_name}  (k={self.k}) …")

        records: list[dict] = []

        for query_title, expected_title in self.golden_pairs:
            try:
                df = model.query(
                    titles=[query_title],
                    source_domain=source_domain,
                    target_domain=target_domain,
                    top_k=self.k,
                    explain=False,
                )
            except Exception as exc:
                print(f"  WARN: '{query_title}' → {exc}")
                records.append({
                    "model":                model_name,
                    "query_title":          query_title,
                    "expected_title":       expected_title,
                    "hit_at_k":             0,
                    "rr":                   0.0,
                    "semantic_ndcg_at_k":   0.0,
                    "returned_titles":      [],
                    "found_at_rank":        None,
                })
                continue

            ranked = df["title"].tolist() if not df.empty else []
            # expected_title is a tuple[str,...]; primary (rank-1) book is [0]
            primary_expected = expected_title[0] if isinstance(expected_title, tuple) else expected_title
            hit = hit_rate_at_k(ranked, primary_expected, self.k)
            rr = reciprocal_rank(ranked, primary_expected, self.k)

            # ── Hybrid NDCG ───────────────────────────────────────────────────
            # Combines hard gold-label bonuses (+2.0/+1.5/+1.0 for gold_1/2/3)
            # with soft semantic similarity (DEFAULT_SEMANTIC_WEIGHT = 0.9).
            # Normalises against a *global* ideal list so a popularity baseline
            # that never surfaces a gold book scores near 0, not ~0.87.
            sem_rels = build_hybrid_relevance_vector(
                ranked, expected_title, self._title_to_emb, self.k
            )
            ideal_rels = build_ideal_relevance_vector(expected_title, self.k)
            sem_ndcg = ndcg_at_k(sem_rels, self.k, ideal_relevances=ideal_rels)

            # Exact match rank (for display / HR/MRR) — check against primary expected book
            found_rank: int | None = next(
                (i for i, t in enumerate(ranked, 1)
                 if t.lower().strip() == primary_expected.lower().strip()),
                None,
            )
            # Partial match fallback for display only
            if found_rank is None:
                exp = primary_expected.lower()
                found_rank = next(
                    (i for i, t in enumerate(ranked, 1)
                     if exp in t.lower() or t.lower() in exp),
                    None,
                )

            status = f"✓ rank {found_rank}" if found_rank else "✗ not found"
            print(f"  {query_title:<40} {status}  sem-NDCG={sem_ndcg:.3f}")

            records.append({
                "model":                model_name,
                "query_title":          query_title,
                "expected_title":       expected_title,
                "hit_at_k":             hit,
                "rr":                   round(rr, 4),
                "semantic_ndcg_at_k":   round(sem_ndcg, 4),
                "returned_titles":      ranked,
                "found_at_rank":        found_rank,
            })

        return pd.DataFrame(records)

    # ── Multi-model comparison ────────────────────────────────────────────────

    def compare(
        self,
        models:        list,
        source_domain: str = "movie",
        target_domain: str = "book",
    ) -> dict[str, pd.DataFrame]:
        """
        Runs ``evaluate_model`` for each model and aggregates results.

        Returns
        -------
        dict with two keys:
            ``"per_query"``  — combined DataFrame of all model × query rows
            ``"summary"``    — one row per model with HR, MRR, Semantic-NDCG averages
        """
        per_query = pd.concat(
            [self.evaluate_model(m, source_domain, target_domain) for m in models],
            ignore_index=True,
        )

        summary = (
            per_query
            .groupby("model", sort=False)
            .agg(
                HR=("hit_at_k",            "mean"),
                MRR=("rr",                  "mean"),
                Semantic_NDCG=("semantic_ndcg_at_k", "mean"),
                n_queries=("query_title",   "count"),
            )
            .round(4)
            .reset_index()
        )
        summary.columns = [
            "Model",
            f"HR@{self.k}",
            f"MRR@{self.k}",
            f"SemanticNDCG@{self.k}",
            "Queries evaluated",
        ]

        return {"per_query": per_query, "summary": summary}

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_report(self, results: dict[str, pd.DataFrame]) -> None:
        """Prints a formatted summary table to stdout."""
        summary = results["summary"]
        per_query = results["per_query"]
        k = self.k
        sep = "═" * 72

        print(f"\n{sep}")
        print("  VIBE MATCHER — EVALUATION REPORT")
        print(f"  Golden pairs evaluated : {len(self.golden_pairs)}")
        print(f"  Metric cutoff          : @{k}")
        print(f"  NDCG variant           : Hybrid (gold-label bonuses + semantic similarity)")
        print(sep)
        print(summary.to_string(index=False))
        print(sep)

        best_idx = summary[f"SemanticNDCG@{k}"].idxmax()
        best_model = summary.loc[best_idx, "Model"]
        best_score = summary.loc[best_idx, f"SemanticNDCG@{k}"]
        print(f"\n  Best SemanticNDCG@{k} → {best_model}  ({best_score:.4f})")

        # Queries where no model found the correct answer
        pivot = per_query.pivot(
            index="query_title", columns="model", values="hit_at_k"
        )
        never_found = pivot[pivot.sum(axis=1) == 0]
        if not never_found.empty:
            print(f"\n  Queries no model answered correctly ({len(never_found)}):")
            for q in never_found.index:
                print(f"    • {q}")

        print(sep)

    def save_report(
        self,
        results:  dict[str, pd.DataFrame],
        path:     str | None = None,
    ) -> None:
        """
        Saves per-query and summary CSVs to *path*.

        Parameters
        ----------
        path:
            Destination for the per-query CSV.
            Defaults to ``RESULTS_DIR/eval_results.csv``.
            The summary is saved alongside as ``eval_results_summary.csv``.
        """
        if path is None:
            path = str(RESULTS_DIR / "eval_results.csv")

        summary_path = path.replace(".csv", "_summary.csv")
        results["per_query"].to_csv(path, index=False)
        results["summary"].to_csv(summary_path, index=False)
        print(f"\n  Saved → {path}")
        print(f"  Saved → {summary_path}")
