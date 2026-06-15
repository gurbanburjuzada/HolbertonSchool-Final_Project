from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from src.matcher.config import RESULTS_DIR
from src.evaluation.evaluator import GOLDEN_PAIRS, Evaluator
from src.matcher.baselines import PopularityBaseline, TFIDFBaseline
from src.evaluation.metrics import ndcg_at_k

warnings.filterwarnings("ignore")


# ── Manual / interactive evaluation ──────────────────────────────────────────

def run_manual_evaluation(
    engine,
    queries:       list[str] | None = None,
    top_k:         int = 3,
    source_domain: str = "movie",
    target_domain: str = "book",
) -> pd.DataFrame:
    """
    Interactive terminal evaluation on a 3-point relevance scale.

    Scale: 2 = strong match  |  1 = partial match  |  0 = mismatch

    Mirrors the "40-50 query-result pairs on a 3-point scale" described in the
    design doc (Section 3.5).  Works in any terminal (or Jupyter/Colab cell).

    Parameters
    ----------
    engine:
        Any model that exposes a ``.query()`` method.
    queries:
        List of source titles to evaluate.  Defaults to the first 10 movies
        in ``GOLDEN_PAIRS``.
    top_k:
        Number of results shown per query.
    source_domain / target_domain:
        Passed through to ``engine.query()``.

    Returns
    -------
    pd.DataFrame
        Columns: ``query``, ``rank``, ``result``, ``relevance``.
        Compute NDCG over the returned labels with ``ndcg_at_k()``.
    """
    if queries is None:
        queries = [pair[0] for pair in GOLDEN_PAIRS[:10]]

    records: list[dict] = []
    print("\n─── Manual Evaluation (enter 0 / 1 / 2 for each result) ───")
    print("  0 = mismatch   1 = partial match   2 = strong match\n")

    for query in queries:
        print(f"\n{'━' * 60}")
        print(f"  Query movie: '{query}'")
        print("━" * 60)

        try:
            df = engine.query(
                titles=[query], source_domain=source_domain,
                target_domain=target_domain, top_k=top_k, explain=False,
            )
        except Exception as exc:
            print(f"  Error: {exc}")
            continue

        for _, row in df.iterrows():
            print(f"\n  #{int(row['rank'])}  {row['title']}")
            while True:
                try:
                    raw = input("       Relevance (0/1/2): ").strip()
                    label = int(raw)
                    if label in (0, 1, 2):
                        break
                    print("       Please enter 0, 1, or 2.")
                except (ValueError, EOFError):
                    label = 0
                    break
            records.append({
                "query":     query,
                "rank":      int(row["rank"]),
                "result":    row["title"],
                "relevance": label,
            })

    df_out = pd.DataFrame(records)
    if not df_out.empty:
        mean_ndcg = (
            df_out.groupby("query")
            .apply(lambda g: ndcg_at_k(
                g.sort_values("rank")["relevance"].tolist(), top_k
            ))
            .mean()
        )
        print(f"\n  Human-labelled NDCG@{top_k}: {mean_ndcg:.4f}")

    return df_out


# ── Convenience runner ────────────────────────────────────────────────────────

def run_full_evaluation(engine, results_dir: Path = RESULTS_DIR) -> dict:
    """
    Convenience function: instantiates all baselines, runs the full evaluation,
    prints the report, and saves CSVs.

    Parameters
    ----------
    engine:
        A loaded ``QueryEngine`` instance (or any model with ``.query()``).
    results_dir:
        Where to write output CSVs.  Defaults to ``RESULTS_DIR``.

    Returns
    -------
    dict with ``"per_query"`` and ``"summary"`` DataFrames.
    """
    pop = PopularityBaseline()
    tfid = TFIDFBaseline()

    ev = Evaluator(golden_pairs=GOLDEN_PAIRS, k=15)
    results = ev.compare([engine, tfid, pop])

    ev.print_report(results)
    ev.save_report(results, path=str(results_dir / "eval_results.csv"))

    return results
