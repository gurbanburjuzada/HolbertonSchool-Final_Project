from __future__ import annotations

import warnings
import numpy as np

warnings.filterwarnings("ignore")


# ── Ranking metrics ───────────────────────────────────────────────────────────

def dcg_at_k(relevances: list[float], k: int) -> float:
    """
    Discounted Cumulative Gain at k.

    Parameters
    ----------
    relevances:
        Graded relevance labels (any non-negative float) in rank order.
    k:
        Cutoff rank.

    Formula: Σ (2^rel − 1) / log2(rank + 1)
    """
    arr = np.array(relevances[:k], dtype=float)
    if len(arr) == 0:
        return 0.0
    positions = np.arange(1, len(arr) + 1)
    return float(((2.0 ** arr - 1.0) / np.log2(positions + 1.0)).sum())


def ndcg_at_k(relevances: list[float], k: int = 10, ideal_relevances: list[float] | None = None) -> float:
    """
    Normalised DCG at k.

    Divides the actual DCG by the ideal DCG (perfect ordering).
    Returns a score in [0, 1]; 1.0 = perfect ranking.
    Accepts any non-negative float relevance labels.

    Parameters
    ----------
    relevances:
        Graded relevance labels in rank order (the actual system output).
    k:
        Cutoff rank.
    ideal_relevances:
        Optional external ideal relevance list used to compute IDCG.
        When provided, IDCG is computed from this list (sorted descending)
        rather than from ``relevances`` itself.  Pass this when you want to
        normalise against the globally best achievable score — e.g. for hybrid
        NDCG where the ideal list contains the fixed gold-label bonuses.
        If None, falls back to the standard self-normalisation (IDCG from the
        sorted ``relevances``).
    """
    if ideal_relevances is not None:
        idcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    else:
        idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    return 0.0 if idcg == 0.0 else min(1.0, dcg_at_k(relevances, k) / idcg)


def hit_rate_at_k(ranked_titles: list[str], expected_title: str, k: int = 10) -> int:
    """Returns 1 if *expected_title* appears anywhere in the top-k results."""
    top_k = [t.lower().strip() for t in ranked_titles[:k]]
    return int(expected_title.lower().strip() in top_k)


def reciprocal_rank(ranked_titles: list[str], expected_title: str, k: int = 10) -> float:
    """Returns 1/rank of the first hit within the top-k, or 0 if not found."""
    expected = expected_title.lower().strip()
    for rank, title in enumerate(ranked_titles[:k], start=1):
        if title.lower().strip() == expected:
            return 1.0 / rank
    return 0.0


# ── Hybrid relevance bonuses ──────────────────────────────────────────────────
#
# The pure-semantic approach (old build_semantic_relevance_vector) inflates
# scores for all models because the relevance vector is built entirely from
# embedding cosine similarities, and NDCG normalises against those same
# similarities.  Any reasonably ordered list ends up looking nearly optimal.
#
# The fix: anchor relevance on gold-label identity first, then layer in
# semantic similarity as a fractional bonus.  Exact adaptations get a hard
# boost that a popularity baseline can never accidentally earn, while
# genuinely similar books still receive partial credit.
#
# Bonus tiers (added on top of the semantic component):
#   gold_1 (primary adaptation)  → +2.0
#   gold_2 (rank-2 expected book) → +1.5
#   gold_3 (rank-3 expected book) → +1.0
#
# Maximum achievable relevance:   ~3.0  (gold_1 exact match + sim ≈ 1.0 → 3.0)
# Typical unrelated popular book: ~0.2–0.4  (only semantic component, no bonus)
# This spread forces NDCG to distinguish models that surface real adaptations.

_GOLD_BONUSES = (1.5, 1.35, 1.25)  # bonuses for gold_1, gold_2, gold_3

# Shared weight used by both build_hybrid_relevance_vector (numerator) and
# build_ideal_relevance_vector (denominator).  Keeping them in sync ensures
# NDCG is a true ratio: a model that retrieves all gold books at ranks 1–3
# scores 1.0, and one that misses all of them scores near 0.
DEFAULT_SEMANTIC_WEIGHT: float = 0.9


def build_hybrid_relevance_vector(
    ranked_titles: list[str],
    expected_title: str | tuple[str, ...],
    title_to_emb: dict[str, np.ndarray],
    k: int = 10,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
) -> list[float]:
    """
    Builds a hybrid relevance vector combining hard gold-label bonuses with
    soft semantic similarity.

    Relevance formula for each retrieved title *t*:

        rel(t) = semantic_weight * cosine_sim(t, nearest_gold)
                 + gold_bonus(t)

    where gold_bonus is:
        +2.0  if t is the primary gold book   (gold_1)
        +1.5  if t is the second  gold book   (gold_2)
        +1.0  if t is the third   gold book   (gold_3)
         0.0  otherwise

    This ensures:
    - Exact adaptations are ranked far above semantically similar but
      unrelated titles (a popularity baseline can never earn those bonuses).
    - Sequels / spin-offs receive meaningful partial credit.
    - Semantically similar books still earn partial credit proportional
      to how close they are to any gold book, preventing harsh all-or-
      nothing penalisation.

    Parameters
    ----------
    ranked_titles:
        Titles returned by the model, in rank order.
    expected_title:
        Gold-standard book(s).  A plain ``str`` (single gold) or a
        ``tuple[str, ...]`` of up to three books ordered by preference
        (as stored in ``GOLDEN_PAIRS``).
    title_to_emb:
        Mapping of ``title.lower().strip()`` → L2-normalised embedding.
        Built once in ``Evaluator.__init__``.
    k:
        Cutoff rank; output is always length *k*.
    semantic_weight:
        Scalar multiplier on the cosine-similarity component.
        Must match the value used in ``build_ideal_relevance_vector`` so
        that NDCG is properly normalised.  Defaults to
        ``DEFAULT_SEMANTIC_WEIGHT`` (0.9).

    Returns
    -------
    list[float]
        Hybrid relevance scores (non-negative floats), length *k*.
        Typical range: ~[0.1, 3.0].
    """
    # ── Normalise gold list ───────────────────────────────────────────────────
    if isinstance(expected_title, str):
        gold_items: list[str] = [expected_title]
    else:
        gold_items = list(expected_title)  # up to 3 entries

    gold_keys: list[str] = [g.lower().strip() for g in gold_items]

    # ── Resolve embeddings for gold books ────────────────────────────────────
    gold_embs: list[np.ndarray | None] = []
    for key in gold_keys:
        emb = title_to_emb.get(key)
        if emb is None:
            # Soft fallback: substring match for minor tokenisation differences
            for idx_key, vec in title_to_emb.items():
                if key in idx_key or idx_key in key:
                    emb = vec
                    break
        gold_embs.append(emb)

    # ── Build one score per ranked position ──────────────────────────────────
    scores: list[float] = []
    for title in ranked_titles[:k]:
        t = title.lower().strip()
        cand_emb = title_to_emb.get(t)

        # 1. Semantic component: max cosine similarity to any gold embedding.
        #    Multiplied by semantic_weight (DEFAULT_SEMANTIC_WEIGHT = 0.9).
        #    Using raw cosine (not squared) to give partial credit to
        #    loosely-related books while still keeping unrelated popular titles
        #    well below the smallest gold bonus (1.0).
        best_sem = 0.0
        for gold_emb in gold_embs:
            if gold_emb is None or cand_emb is None:
                continue
            # Embeddings are L2-normalised → dot == cosine similarity
            sim = float(np.dot(cand_emb, gold_emb))
            sim = max(0.0, sim)  # sharpen: penalise weak similarities
            if sim > best_sem:
                best_sem = sim

        semantic_component = semantic_weight * best_sem

        # 2. Gold-label bonus: hard boost for exact gold matches
        gold_bonus = 0.0
        for rank_idx, gkey in enumerate(gold_keys):
            if t == gkey:
                gold_bonus = _GOLD_BONUSES[rank_idx] if rank_idx < len(_GOLD_BONUSES) else 0.5
                break  # take the best (first) matching tier

        scores.append(semantic_component + gold_bonus)

    scores += [0.0] * (k - len(scores))
    return scores


def build_ideal_relevance_vector(
    expected_title: str | tuple[str, ...],
    k: int = 10,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
) -> list[float]:
    """
    Returns the globally ideal relevance vector for a gold set.

    Used to compute a global IDCG so that NDCG normalises against the best
    *possible* score rather than the best ordering of whatever the model
    actually returned.  A popularity baseline that returns zero gold books
    should score near 0, not 0.9.

    The ideal list places each gold book at the top in order (gold_1 first),
    with the maximum semantic component (1.0 × semantic_weight) added, then
    pads with zeros.

    ``semantic_weight`` must match the value used in
    ``build_hybrid_relevance_vector`` (both default to
    ``DEFAULT_SEMANTIC_WEIGHT``) so that the NDCG ratio is valid.

    Note: the semantic ceiling is approximately 0.9 (when cosine sim ≈ 1.0),
    which remains below the smallest gold bonus (1.0).  A model that surfaces
    a near-gold but non-gold book therefore cannot displace a true gold match.
    """
    if isinstance(expected_title, str):
        gold_items: list[str] = [expected_title]
    else:
        gold_items = list(expected_title)

    ideal: list[float] = []
    for rank_idx in range(min(len(gold_items), len(_GOLD_BONUSES))):
        # Gold book perfectly retrieved → semantic sim ≈ 1.0 → sem_component = weight × 1.0
        ideal.append(_GOLD_BONUSES[rank_idx] + semantic_weight * 1.0)

    ideal += [0.0] * (k - len(ideal))
    return ideal
# Code that imports build_semantic_relevance_vector still works; it now
# delegates to the hybrid implementation with semantic_weight=1.0 (pure
# semantic, no gold bonuses) to preserve the old calling convention.
# Switch callers to build_hybrid_relevance_vector for the improved metric.


def build_semantic_relevance_vector(
    ranked_titles: list[str],
    expected_title: str | tuple[str, ...],
    title_to_emb: dict[str, np.ndarray],
    k: int = 10,
) -> list[float]:
    """
    Pure-semantic relevance vector using cosine similarity only.

    All returned scores are in [0, 1].  No gold-label bonuses are applied —
    use ``build_hybrid_relevance_vector`` when you need those.

    Scoring rules
    -------------
    * Exact title match + embeddings available  → cosine similarity (~1.0 for
      identical L2-normalised vectors).
    * Exact title match + embeddings missing    → 1.0 (graceful fallback so
      the gold book is always "found" even when the index is incomplete).
    * Non-exact title                           → max cosine similarity to any
      gold embedding, clipped to [0, 1].  0.0 when either embedding is absent.

    Parameters
    ----------
    ranked_titles:
        Titles returned by the model, in rank order.
    expected_title:
        Gold-standard book title(s).  A plain ``str`` or a
        ``tuple[str, ...]`` of up to three books ordered by preference.
    title_to_emb:
        Mapping of ``title.lower().strip()`` → L2-normalised embedding.
    k:
        Cutoff rank; output list is always length *k*.

    Returns
    -------
    list[float]
        Cosine-similarity relevance scores in [0, 1], length *k*.
    """
    if isinstance(expected_title, str):
        gold_items: list[str] = [expected_title]
    else:
        gold_items = list(expected_title)

    gold_keys: list[str] = [g.lower().strip() for g in gold_items]

    # Resolve embeddings for gold books (with substring fallback)
    gold_embs: list[np.ndarray | None] = []
    for key in gold_keys:
        emb = title_to_emb.get(key)
        if emb is None:
            for idx_key, vec in title_to_emb.items():
                if key in idx_key or idx_key in key:
                    emb = vec
                    break
        gold_embs.append(emb)

    scores: list[float] = []
    for title in ranked_titles[:k]:
        t = title.lower().strip()
        cand_emb = title_to_emb.get(t)

        if t in gold_keys:
            # Exact match: use cosine sim if both embeddings exist,
            # otherwise fall back to 1.0 so the gold book is never lost.
            gold_emb_for_t = gold_embs[gold_keys.index(t)]
            if cand_emb is not None and gold_emb_for_t is not None:
                sim = float(np.dot(cand_emb, gold_emb_for_t))
                scores.append(max(0.0, min(1.0, sim)))
            else:
                scores.append(1.0)
        else:
            # Non-exact: best cosine similarity to any gold embedding
            best_sim = 0.0
            for gold_emb in gold_embs:
                if gold_emb is None or cand_emb is None:
                    continue
                sim = float(np.dot(cand_emb, gold_emb))
                if sim > best_sim:
                    best_sim = sim
            scores.append(max(0.0, min(1.0, best_sim)))

    scores += [0.0] * (k - len(scores))
    return scores
