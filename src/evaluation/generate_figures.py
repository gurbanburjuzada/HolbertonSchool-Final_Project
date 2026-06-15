"""
generate_figures.py
====================
VibeMatcher Evaluation — Figure Generator
------------------------------------------
Run from the project root:
    python generate_figures.py

Reads:
    results/eval_results.csv          (per-query, per-model rows)
    results/eval_results_summary.csv  (aggregate summary)

Writes 7 PNG figures into:
    figures/

Dependencies (already in requirements.txt):
    pandas, matplotlib, numpy, seaborn
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.matcher.config import RESULTS_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DETAIL_CSV = RESULTS_DIR / "eval_results.csv"
SUMMARY_CSV = RESULTS_DIR / "eval_results_summary.csv"

# ── Palette & style ────────────────────────────────────────────────────────────
# Three purposeful colours — one per model, consistent across all figures.
MODEL_COLORS = {
    "VibeMatcher (Semantic + Genre + Mood)": "#5B8FF9",   # clear blue  — the hero model
    "TF-IDF Baseline":                       "#F6BD16",   # warm amber  — keyword baseline
    "Popularity Baseline":                   "#E86452",   # muted red   — dumb baseline
}

MODEL_SHORT = {
    "VibeMatcher (Semantic + Genre + Mood)": "VibeMatcher",
    "TF-IDF Baseline":                       "TF-IDF",
    "Popularity Baseline":                   "Popularity",
}

FONT_TITLE = dict(fontsize=13, fontweight="bold", color="#1a1a2e")
FONT_AXIS = dict(fontsize=10, color="#333333")
FONT_TICK = dict(labelsize=9, colors="#555555")
GRID_STYLE = dict(color="#e0e0e0", linewidth=0.7, linestyle="--")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
})
k = 15

# ── Load data ─────────────────────────────────────────────────────────────────

detail = pd.read_csv(DETAIL_CSV)
summary = pd.read_csv(SUMMARY_CSV)

# Normalise column names
detail.columns = detail.columns.str.strip()
summary.columns = summary.columns.str.strip()

# Parse found_at_rank to numeric (NaN = miss)
detail["found_at_rank"] = pd.to_numeric(detail["found_at_rank"], errors="coerce")

models_ordered = [
    "VibeMatcher (Semantic + Genre + Mood)",
    "TF-IDF Baseline",
    "Popularity Baseline",
]

# Short labels for tick annotations
detail["model_short"] = detail["model"].map(MODEL_SHORT)
summary["model_short"] = summary["Model"].map(MODEL_SHORT)
summary = summary.set_index("Model")

print("✓ Data loaded.")
print(f"  {len(detail)} per-query rows  |  {detail['model'].nunique()} models  "
      f"|  {detail['query_title'].nunique()} queries\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Overall Model Comparison (grouped bar chart)
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_overall_comparison():
    metrics = [f"HR@{k}", f"MRR@{k}", f"SemanticNDCG@{k}"]
    col_map = {f"HR@{k}": f"HR@{k}", f"MRR@{k}": f"MRR@{k}", f"SemanticNDCG@{k}": f"SemanticNDCG@{k}"}
    display_name = {f"HR@{k}": f"Hit Rate\n@{k}", f"MRR@{k}": f"MRR\n@{k}",
                    f"SemanticNDCG@{k}": f"Semantic\nNDCG@{k}"}

    n_metrics = len(metrics)
    n_models = len(models_ordered)
    x = np.arange(n_metrics)
    width = 0.22
    offsets = np.linspace(-(n_models - 1) * width / 2,
                          (n_models - 1) * width / 2, n_models)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor("white")

    for i, model in enumerate(models_ordered):
        values = [summary.loc[model, col_map[m]] for m in metrics]
        bars = ax.bar(
            x + offsets[i], values, width,
            color=MODEL_COLORS[model],
            label=MODEL_SHORT[model],
            zorder=3,
            linewidth=0,
        )
        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7.5, color="#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([display_name[m] for m in metrics], fontsize=10.5)
    ax.set_ylabel("Score", **FONT_AXIS)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, **GRID_STYLE)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", **FONT_TICK)
    ax.tick_params(axis="x", bottom=False)

    ax.set_title("VibeMatcher vs. Baselines — Aggregate Metrics (k = 15)",
                 pad=14, **FONT_TITLE)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left",
              bbox_to_anchor=(0.01, 0.97))

    fig.tight_layout()
    out = FIGURES_DIR / "fig1_overall_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ fig1  →  {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Per-query SemanticNDCG@15
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_per_query_ndcg():
    queries = detail["query_title"].unique()
    queries_sorted = sorted(queries, key=lambda q: (
        -detail.loc[
            (detail["query_title"] == q) &
            (detail["model"] == "VibeMatcher (Semantic + Genre + Mood)"),
            "semantic_ndcg_at_k"
        ].mean()
    ))

    n_q = len(queries_sorted)
    x = np.arange(n_q)
    width = 0.26
    offsets = np.array([-width, 0.0, width])

    fig, ax = plt.subplots(figsize=(16, 5.5))

    for i, model in enumerate(models_ordered):
        sub = detail[detail["model"] == model].set_index("query_title")
        values = [sub.loc[q, "semantic_ndcg_at_k"] if q in sub.index else 0.0
                  for q in queries_sorted]
        ax.bar(x + offsets[i], values, width,
               color=MODEL_COLORS[model],
               label=MODEL_SHORT[model],
               zorder=3, linewidth=0)

    ax.set_xticks(x)
    ax.set_xticklabels(queries_sorted, rotation=42, ha="right",
                        fontsize=8, color="#333333")
    ax.set_ylabel(f"Semantic NDCG@{k}", **FONT_AXIS)
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, **GRID_STYLE)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", **FONT_TICK)
    ax.tick_params(axis="x", bottom=False)
    ax.set_title(f"Semantic NDCG@{k} per Query — All Models",
                 pad=12, **FONT_TITLE)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    out = FIGURES_DIR / "fig2_per_query_ndcg.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ fig2  →  {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Rank-of-First-Hit Distribution (horizontal histogram)
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_rank_distribution():
    """For each model, show how many hits land at each rank position (1–15)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

    for ax, model in zip(axes, models_ordered):
        sub = detail[detail["model"] == model]
        ranks = sub["found_at_rank"].dropna().astype(int)
        counts = ranks.value_counts().reindex(range(1, 16), fill_value=0)

        ax.barh(
            counts.index[::-1], counts.values[::-1],
            color=MODEL_COLORS[model], linewidth=0, zorder=3
        )
        # Annotate counts
        for rank, cnt in counts.items():
            if cnt > 0:
                ax.text(cnt + 0.05, 16 - rank,
                        str(cnt), va="center", fontsize=9, color="#333333")

        ax.set_yticks(range(1, 16))
        ax.set_yticklabels([f"Rank {r}" for r in range(1, 16)],
                            fontsize=8.5, color="#444444")
        ax.set_xlabel("# Queries", **FONT_AXIS)
        ax.set_title(MODEL_SHORT[model], fontsize=11, fontweight="bold",
                     color=MODEL_COLORS[model], pad=8)
        ax.xaxis.grid(True, **GRID_STYLE)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", **FONT_TICK)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False)

    fig.suptitle("Rank of First Hit  (where the gold book appeared in the top-15)",
                 y=1.02, **FONT_TITLE)
    fig.tight_layout()
    out = FIGURES_DIR / "fig3_rank_distribution.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ fig3  →  {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Hit / Miss Matrix  (model × query)
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_hit_miss_matrix():
    pivot = detail.pivot_table(
        index="model", columns="query_title",
        values="hit_at_k", aggfunc="max"
    )
    # Reorder rows
    pivot = pivot.reindex(models_ordered)

    # Shorten row labels
    pivot.index = [MODEL_SHORT[m] for m in pivot.index]

    # Sort columns by VibeMatcher score so hits cluster left
    col_order = pivot.loc["VibeMatcher"].sort_values(ascending=False).index
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(16, 3.5))

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "hm", ["#f5f5f5", "#5B8FF9"]
    )
    sns.heatmap(
        pivot.astype(float),
        ax=ax,
        cmap=cmap,
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor="#dddddd",
        cbar_kws={"label": "Hit (1) / Miss (0)", "shrink": 0.6},
        annot=True, fmt=".0f",
        annot_kws={"fontsize": 8, "color": "#222222"},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=42,
                        ha="right", fontsize=8, color="#333333")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                        fontsize=9.5, color="#222222")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Hit / Miss Matrix — Top-15 Results per Query",
                 pad=12, **FONT_TITLE)

    fig.tight_layout()
    out = FIGURES_DIR / "fig4_hit_miss_matrix.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ fig4  →  {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Metric Uplift: VibeMatcher vs. Baselines
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_metric_uplift():
    """
    Show % relative improvement of VibeMatcher over each baseline
    for each metric.
    """
    baselines = ["TF-IDF Baseline", "Popularity Baseline"]
    metrics = [f"HR@{k}", f"MRR@{k}", f"SemanticNDCG@{k}"]
    metric_lab = {f"HR@{k}": f"HR@{k}", f"MRR@{k}": f"MRR@{k}",
                  f"SemanticNDCG@{k}": f"Semantic NDCG@{k}"}

    vm = summary.loc["VibeMatcher (Semantic + Genre + Mood)"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, baseline in zip(axes, baselines):
        bl = summary.loc[baseline]
        uplifts = []
        for m in metrics:
            bl_val = bl[m]
            vm_val = vm[m]
            if bl_val == 0:
                uplift = float("inf") if vm_val > 0 else 0.0
            else:
                uplift = (vm_val - bl_val) / bl_val * 100
            uplifts.append(uplift)

        # Cap Inf for display
        display_uplifts = [min(u, 2500) for u in uplifts]

        colors_bar = ["#5B8FF9"] * len(metrics)
        bars = ax.barh(
            [metric_lab[m] for m in metrics],
            display_uplifts,
            color=colors_bar,
            height=0.45,
            zorder=3,
            linewidth=0,
        )

        for bar, raw, disp in zip(bars, uplifts, display_uplifts):
            label = f"+{raw:.0f}%" if not np.isinf(raw) else "∞%"
            ax.text(
                disp + 15, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10, color="#222222",
                fontweight="bold",
            )

        ax.axvline(0, color="#aaaaaa", linewidth=1)
        ax.xaxis.grid(True, **GRID_STYLE)
        ax.set_axisbelow(True)
        ax.set_xlabel("Relative improvement (%)", **FONT_AXIS)
        ax.set_title(
            f"vs. {MODEL_SHORT[baseline]}",
            fontsize=11, fontweight="bold",
            color=MODEL_COLORS[baseline], pad=8,
        )
        ax.tick_params(axis="y", **FONT_TICK)
        ax.tick_params(axis="x", **FONT_TICK)
        ax.spines["bottom"].set_visible(False)

    fig.suptitle("VibeMatcher Relative Uplift over Baselines",
                 y=1.01, **FONT_TITLE)
    fig.tight_layout()
    out = FIGURES_DIR / "fig5_metric_uplift.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ fig5  →  {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — NDCG Distribution (violin + strip)
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_ndcg_distribution():
    """Violin + individual-point strip chart of SemanticNDCG per model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    palette = {MODEL_SHORT[m]: MODEL_COLORS[m] for m in models_ordered}
    detail["Model"] = detail["model"].map(MODEL_SHORT)

    order = [MODEL_SHORT[m] for m in models_ordered]

    vp = sns.violinplot(
        data=detail,
        x="Model", y="semantic_ndcg_at_k",
        order=order,
        hue="Model", hue_order=order,
        palette=palette,
        inner=None,
        linewidth=1.2,
        cut=0,
        legend=False,
        ax=ax,
    )
    # Make violins semi-transparent
    for pc in ax.collections:
        pc.set_alpha(0.45)

    sns.stripplot(
        data=detail,
        x="Model", y="semantic_ndcg_at_k",
        order=order,
        hue="Model", hue_order=order,
        palette=palette,
        size=5, alpha=0.8, jitter=True,
        legend=False,
        ax=ax, zorder=4,
    )

    # Median line annotation
    for i, model_short in enumerate(order):
        med = detail.loc[detail["Model"] == model_short, "semantic_ndcg_at_k"].median()
        ax.hlines(med, i - 0.35, i + 0.35,
                   colors="#222222", linewidths=2, zorder=5)
        ax.text(i + 0.38, med, f"med={med:.3f}",
                va="center", fontsize=8, color="#333333")

    ax.yaxis.grid(True, **GRID_STYLE)
    ax.set_axisbelow(True)
    ax.set_xlabel("")
    ax.set_ylabel(f"Semantic NDCG@{k}", **FONT_AXIS)
    ax.tick_params(axis="x", bottom=False, labelsize=10)
    ax.tick_params(axis="y", **FONT_TICK)
    ax.set_title(f"Distribution of Semantic NDCG@{k} across Queries",
                 pad=12, **FONT_TITLE)

    fig.tight_layout()
    out = FIGURES_DIR / "fig6_ndcg_distribution.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ fig6  →  {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Per-query Reciprocal Rank (only queries with ≥1 hit)
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_per_query_rr():
    """
    Dot-plot of reciprocal rank per query, for queries where at least
    one model returned a hit.  Makes rank quality easy to compare.
    """
    # Queries where ANY model got a hit
    hit_queries = (detail[detail["hit_at_k"] == 1]["query_title"].unique())

    sub = detail[detail["query_title"].isin(hit_queries)].copy()
    # Sort queries by VibeMatcher RR descending
    order_map = (
        sub[sub["model"] == "VibeMatcher (Semantic + Genre + Mood)"]
        .set_index("query_title")["rr"]
        .to_dict()
    )
    sorted_queries = sorted(hit_queries,
                             key=lambda q: order_map.get(q, 0), reverse=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    y_pos  = np.arange(len(sorted_queries))
    width  = 0.22
    offsets = np.array([-width, 0.0, width])

    for i, model in enumerate(models_ordered):
        ms = sub[sub["model"] == model].set_index("query_title")
        vals = [ms.loc[q, "rr"] if q in ms.index else 0.0
                for q in sorted_queries]
        ax.barh(
            y_pos + offsets[i], vals, width,
            color=MODEL_COLORS[model],
            label=MODEL_SHORT[model],
            zorder=3, linewidth=0,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_queries, fontsize=9, color="#333333")
    ax.set_xlabel("Reciprocal Rank  (1/rank of first hit, 0 = miss)", **FONT_AXIS)
    ax.set_xlim(0, 1.15)
    ax.xaxis.grid(True, **GRID_STYLE)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", **FONT_TICK)
    ax.tick_params(axis="y", left=False)
    ax.spines["left"].set_visible(False)
    ax.invert_yaxis()  # Best at top

    # Reference lines
    for rr_val, label in [(1.0, "Rank 1"), (0.5, "Rank 2"),
                           (0.333, "Rank 3"), (0.1, "Rank 10")]:
        ax.axvline(rr_val, color="#cccccc", linewidth=0.9,
                    linestyle=":", zorder=1)
        ax.text(rr_val, -0.8, label, ha="center",
                fontsize=7.5, color="#999999")

    ax.set_title(
        "Reciprocal Rank per Query  (queries with ≥ 1 hit across any model)",
        pad=12, **FONT_TITLE,
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    out = FIGURES_DIR / "fig7_per_query_rr.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ fig7  →  {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures …\n")
    fig1_overall_comparison()
    fig2_per_query_ndcg()
    fig3_rank_distribution()
    fig4_hit_miss_matrix()
    fig5_metric_uplift()
    fig6_ndcg_distribution()
    fig7_per_query_rr()
    print(f"\nAll 7 figures saved to  {FIGURES_DIR.resolve()}/")
