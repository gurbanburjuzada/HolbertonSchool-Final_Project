"""
app.py — VibeMatcher Streamlit frontend.

Run from the Holberton_Final/ directory:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import streamlit as st
from pathlib import Path
from src.matcher.engine import QueryEngine
from src.matcher.config import INDEX_DIR, GEMINI_API_KEY

# ── SSL fix: force Python to use certifi's certificate bundle instead of the
# OS cert store, which can be broken/incomplete on some Windows setups and
# causes "[ASN1: NOT_ENOUGH_DATA]" errors when connecting to the Gemini API.
import os
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# Ensure src/ is importable when invoked as: streamlit run app.py
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="VibeMatcher",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3                  { font-family: 'Sora', sans-serif !important; }

/* Hero */
.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: #1e293b;
    margin-bottom: 0;
}
.hero-sub {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.4rem;
    margin-bottom: 2rem;
}
.wordmark {
    font-size: 0.72rem;
    color: #94a3b8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
}
.vibe-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #10b981);
    display: inline-block;
    margin-right: 6px;
    vertical-align: middle;
}

/* Result cards */
.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s ease;
}
.result-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.10); }

.rank-badge {
    display: inline-block;
    background: #f1f5f9;
    color: #475569;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 9px;
    border-radius: 20px;
    margin-bottom: 0.5rem;
}
.result-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0.25rem 0 0.5rem;
}

/* Similarity bar */
.sim-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.5rem 0;
}
.sim-bar-bg {
    flex: 1;
    height: 6px;
    background: #e2e8f0;
    border-radius: 3px;
    overflow: hidden;
}
.sim-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #6366f1, #10b981);
}
.sim-label {
    font-size: 0.88rem;
    font-weight: 700;
    color: #4f46e5;
    min-width: 68px;
    text-align: right;
}

/* Sentiment */
.sent-match { color: #10b981; font-size: 0.75rem; font-weight: 500; }
.sent-miss  { color: #f59e0b; font-size: 0.75rem; font-weight: 500; }

/* Genre tags */
.genre-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin: 0.5rem 0;
}
.genre-tag {
    display: inline-block;
    background: #ede9fe;
    color: #5b21b6;
    font-size: 0.70rem;
    font-weight: 500;
    padding: 2px 9px;
    border-radius: 20px;
    letter-spacing: 0.02em;
}

/* Mood tags */
.mood-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin: 0.35rem 0;
}
.mood-tag {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    font-size: 0.68rem;
    font-weight: 500;
    padding: 2px 9px;
    border-radius: 20px;
    letter-spacing: 0.02em;
}

/* Explanation */
.explanation {
    font-size: 0.87rem;
    color: #475569;
    border-left: 3px solid #6366f1;
    padding-left: 0.75rem;
    margin-top: 0.75rem;
    font-style: italic;
    line-height: 1.5;
}

/* Footer */
.footer {
    text-align: center;
    color: #cbd5e1;
    font-size: 0.76rem;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #f1f5f9;
}
</style>
""", unsafe_allow_html=True)


# ── Engine — loaded once, cached for the lifetime of the Streamlit session ────
@st.cache_resource(show_spinner="Loading VibeMatcher index… (first run may take ~30 s)")
def load_engine() -> QueryEngine:
    return QueryEngine(index_dir=INDEX_DIR)


# ── Result card ───────────────────────────────────────────────────────────────
def render_card(row: dict, target_label: str) -> None:
    import html as _html

    sim_pct = int(row["similarity"] * 100)
    bar_width = min(sim_pct, 100)
    genres = row.get("genres") or []
    moods = row.get("mood_tags") or []

    genre_html = (
        "".join(f'<span class="genre-tag">{_html.escape(g.title())}</span>' for g in genres)
        if genres
        else '<span style="color:#94a3b8;font-size:0.75rem;">no genres tagged</span>'
    )

    mood_html = (
        "".join(f'<span class="mood-tag">✦ {_html.escape(m)}</span>' for m in moods)
        if moods
        else ""
    )
    mood_row_html = (
        f'<div class="mood-row">{mood_html}</div>'
        if mood_html
        else ""
    )

    sent = row.get("sentiment_match")
    if sent is True:
        sent_html = '<span class="sent-match">✓ mood match</span>'
    elif sent is False:
        sent_html = '<span class="sent-miss">~ different tone</span>'
    else:
        sent_html = ""

    explanation = (row.get("explanation") or "").strip()
    explanation_html = (
        f'<div class="explanation">💡 {explanation}</div>'
        if explanation and explanation != "(explanation unavailable)"
        else ""
    )

    st.markdown(f"""
    <div class="result-card">
        <span class="rank-badge">#{row['rank']} &nbsp;·&nbsp; {target_label}</span>
        <div class="result-title">{row['title']}</div>
        <div class="sim-row">
            <div class="sim-bar-bg">
                <div class="sim-bar-fill" style="width:{bar_width}%"></div>
            </div>
            <span class="sim-label">{sim_pct}%</span>
            {sent_html}
        </div>
        <div class="genre-row">{genre_html}</div>
        {mood_row_html}
        {explanation_html}
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:

    # Hero header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:1rem;">
        <span class="vibe-dot"></span>
        <span class="wordmark">VibeMatcher</span>
    </div>
    <div class="hero-title">Find your next<br>favourite.</div>
    <div class="hero-sub">
        Type a movie and discover books with the same vibe — or the other way around.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([2.5, 1])

    # ─────────────────────────────
    # LEFT SIDE
    # ─────────────────────────────

    with left:

        col_title, col_domain = st.columns([3, 1])

        with col_title:
            title_input = st.text_input(
                "Title",
                placeholder="e.g. The Shining, Dune…",
                label_visibility="collapsed",
            )

        with col_domain:
            source_domain = st.selectbox(
                "Domain",
                options=["Movie", "Book"],
                label_visibility="collapsed",
            )

        search = st.button(
            "Find Matches ->",
            type="primary",
            use_container_width=True
        )



    # ─────────────────────────────
    # RIGHT SIDE
    # ─────────────────────────────

    with right:

        with st.expander("＋ Blend two titles", expanded=True):
            title2 = st.text_input(
                "Second title",
                placeholder="e.g. Arrival",
                label_visibility="collapsed",
            )

            w1 = st.slider(
                "Weight of first title",
                min_value=0.1,
                max_value=0.9,
                value=0.7,
                step=0.05,
            )

        with st.expander("⚙️ Options", expanded=True):
            top_k = st.slider(
                "Number of recommendations",
                3,
                15,
                5
            )

            genre_boost = st.toggle(
                "Genre boost",
                value=True
            )

            sentiment = st.toggle(
                "Sentiment filter",
                value=True
            )

            explain = st.toggle(
                "AI explanations (Gemini)",
                value=False
            )
    if not search:
        return

    # Validate
    if not title_input.strip():
        st.error("Please enter at least one title.")
        return

    # Build title list + weights
    titles = [title_input.strip()]
    weights = [1.0]
    if title2.strip():
        titles.append(title2.strip())
        weights = [w1, round(1.0 - w1, 2)]

    # Run query
    engine = load_engine()
    target_label = "Book" if source_domain == "Movie" else "Movie"

    with st.spinner(f"Searching for {target_label.lower()} matches…"):
        try:
            results = engine.query(
                titles=titles,
                source_domain=source_domain.lower(),
                weights=weights,
                top_k=top_k,
                apply_genre_boost=genre_boost,
                apply_sentiment_filter=sentiment,
                explain=explain,
            )
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.stop()

    if results.empty:
        st.info(
            "No results returned. The title may not be in the index — "
            "try a slightly different spelling."
        )
        return

    # ── Results header ────────────────────────────────────────────────────────
    blend_suffix = f" + *{title2.strip()}*" if title2.strip() else ""
    st.markdown(f"""
    <div style="margin:1.6rem 0 1rem;">
        <span style="font-size:0.72rem;color:#94a3b8;letter-spacing:0.08em;
                     text-transform:uppercase;font-weight:600;">Results</span><br>
        <span style="font-family:'Sora',sans-serif;font-size:1.35rem;
                     font-weight:600;color:#1e293b;">
            {len(results)} {target_label}s matching
            <em style="color:#4f46e5;">{title_input.strip()}{blend_suffix}</em>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Render each card
    results_left, results_right = st.columns(2)

    for i, (_, row) in enumerate(results.iterrows()):
        if i % 2 == 0:
            with results_left:
                render_card(row.to_dict(), target_label)
        else:
            with results_right:
                render_card(row.to_dict(), target_label)

    # Footer
    st.markdown(
        '<div class="footer">'
        "VibeMatcher &nbsp;·&nbsp; mxbai-embed-large-v1 &nbsp;·&nbsp; "
        "Semantic retrieval + Genre Jaccard re-ranking"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
