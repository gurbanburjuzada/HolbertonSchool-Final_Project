"""
app.py — VibeMatcher Streamlit frontend.

Run from the Holberton_Final/ directory:
    streamlit run app.py
"""
from __future__ import annotations

# ── SSL fix: force Python to use certifi's certificate bundle instead of the
# OS cert store, which can be broken/incomplete on some Windows setups and
# causes "[ASN1: NOT_ENOUGH_DATA]" errors when connecting to the Gemini API.
# Fix: explicitly overwrite these vars whenever they're missing OR empty.

import os
import certifi
import sys
import pandas as pd
import streamlit as st
from pathlib import Path
from src.matcher.engine import QueryEngine
from src.matcher.config import INDEX_DIR, GEMINI_API_KEY
from src.matcher.index_download import ensure_index_files

_CERT_BUNDLE = certifi.where()
for _var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
    if not os.environ.get(_var):          # catches both unset AND ""
        os.environ[_var] = _CERT_BUNDLE

# SSL_CERT_DIR="" (empty capath) triggers its own SSL error
# ([X509: INVALID_DIRECTORY]), so just drop it if it's empty.
if os.environ.get("SSL_CERT_DIR") == "":
    del os.environ["SSL_CERT_DIR"]


# Ensure src/ is importable when invoked as: streamlit run app.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Index files: download from HF Dataset (HF_INDEX_REPO) if not already
# present locally. No-op if index/ already has the 4 required files
# (e.g. local development). See src/matcher/index_download.py for setup.
ensure_index_files()


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
def _safe_list(val) -> list[str]:
    """Return a clean list of plain strings from any genre/mood value.

    Handles: None, empty string, proper list, numpy array, stringified list
    (e.g. "['fantasy', 'sci-fi']"), and silently drops any item that contains
    HTML markup so a stale index with pre-rendered HTML fragments never leaks
    into the UI.
    """
    import ast as _ast
    if val is None:
        return []
    # String — could be a stringified list or a single genre name
    if isinstance(val, str):
        val = val.strip()
        if not val or val.lower() in ("nan", "none", "[]"):
            return []
        if val.startswith("["):
            try:
                parsed = _ast.literal_eval(val)
                if isinstance(parsed, (list, tuple)):
                    val = parsed
            except (ValueError, SyntaxError):
                pass
        if isinstance(val, str):  # still a string after attempted parse
            return [val] if "<" not in val else []
    # List / tuple / numpy array
    if hasattr(val, "__iter__"):
        return [
            str(item)
            for item in val
            if item is not None and str(item).strip() and "<" not in str(item)
        ]
    return []


def render_card(row: dict, target_label: str) -> None:
    import html as _html

    sim_pct = int(row["similarity"] * 100)
    bar_width = min(sim_pct, 100)
    genres = _safe_list(row.get("genres"))
    moods  = _safe_list(row.get("mood_tags"))

    genre_html = (
        "".join(f'<span class="genre-tag">{_html.escape(str(g).title())}</span>' for g in genres)
        if genres
        else '<span style="color:#94a3b8;font-size:0.75rem;">no genres tagged</span>'
    )

    mood_html = (
        "".join(f'<span class="mood-tag">✦ {_html.escape(str(m))}</span>' for m in moods)
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

    explanation = _html.escape((row.get("explanation") or "").strip())
    explanation_html = (
        f'<div class="explanation">💡 {explanation}</div>'
        if explanation and explanation != _html.escape("(explanation unavailable)")
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
@st.cache_data(show_spinner=False)
def cached_query(
    titles: tuple[str, ...],
    source_domain: str,
    weights: tuple[float, ...],
    top_k: int,
    apply_genre_boost: bool,
    apply_sentiment_filter: bool,
    explain: bool,
) -> pd.DataFrame:
    """Thin cacheable wrapper around engine.query().

    ``@st.cache_data`` keys on the arguments, so identical searches
    (including Streamlit reruns from UI interaction) hit local memory
    instead of calling Gemini again.
    """
    engine = load_engine()
    return engine.query(
        titles=list(titles),
        source_domain=source_domain,
        weights=list(weights),
        top_k=top_k,
        apply_genre_boost=apply_genre_boost,
        apply_sentiment_filter=apply_sentiment_filter,
        explain=explain,
    )


def main() -> None:

    # Hero header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:1rem;">
        <span class="vibe-dot"></span>
        <span class="wordmark">VibeMatcher</span>
    </div>
    <div class="hero-title">Find your next<br>obsession.</div>
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
    target_label = "Book" if source_domain == "Movie" else "Movie"

    with st.spinner(f"Searching for {target_label.lower()} matches…"):
        try:
            results = cached_query(
                titles=tuple(titles),
                source_domain=source_domain.lower(),
                weights=tuple(weights),
                top_k=top_k,
                apply_genre_boost=genre_boost,
                apply_sentiment_filter=sentiment,
                explain=explain,
            )
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.stop()

    if explain and "explanation" in results.columns:
        failed = (results["explanation"] == "(explanation unavailable)").sum()
        if failed > 0:
            st.warning(
                f"Gemini explanations failed for {failed}/{len(results)} result(s) "
                "after 3 retries — showing recommendations without them. "
                "This is usually transient (rate limit or response formatting); try again in a moment."
            )

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
