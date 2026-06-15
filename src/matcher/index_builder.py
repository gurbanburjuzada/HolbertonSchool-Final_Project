from __future__ import annotations


import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.matcher.config import MODEL_NAME, SENTIMENT_THRESHOLD, INDEX_DIR, BATCH_SIZE
from src.matcher.preprocessing import _clean_title, _parse_genres


# ── Mood-tag helper ───────────────────────────────────────────────────────────

#: Closed vocabulary that keeps Gemini answers consistent and comparable.
MOOD_VOCAB = frozenset({
    "melancholic", "hopeful", "tragic", "dark", "uplifting",
    "psychological", "heartwarming", "nostalgic", "tense", "whimsical",
    "suspenseful", "romantic", "adventurous", "eerie", "bittersweet",
    "inspiring", "humorous", "gritty",
})

_MOOD_KEYWORDS: dict[str, list[str]] = {
    "melancholic":   ["sad", "grief", "loss", "sorrow", "despair",
                      "mourning", "lonely", "melancholy", "depression", "weep"],
    "hopeful":       ["hope", "redemption", "second chance", "new beginning",
                      "optimism", "faith", "future", "overcome", "possibility"],
    "tragic":        ["tragedy", "tragic", "death", "dying", "fatal",
                      "doomed", "sacrifice", "devastating", "loss", "fallen"],
    "dark":          ["dark", "sinister", "evil", "corrupt", "bleak",
                      "disturbing", "cruel", "brutal", "violent", "oppressive"],
    "uplifting":     ["uplifting", "triumph", "victory", "joy", "celebrate",
                      "succeed", "rise", "overcome", "positive", "heartened"],
    "psychological": ["psychological", "mind", "identity", "obsession",
                      "paranoia", "perception", "delusion", "trauma",
                      "manipulation", "unreliable", "psyche"],
    "heartwarming":  ["heartwarming", "family", "friendship", "bond",
                      "kindness", "community", "caring", "togetherness", "warm"],
    "nostalgic":     ["nostalgia", "nostalgic", "childhood", "memory",
                      "reminisce", "coming of age", "youth", "remember",
                      "past", "simpler times"],
    "tense":         ["tension", "tense", "danger", "threat", "escape",
                      "chase", "confrontation", "conflict", "stakes",
                      "edge of your seat", "nerve"],
    "whimsical":     ["whimsical", "magical", "fantasy", "wonder",
                      "imaginative", "fairy", "enchant", "playful",
                      "surreal", "whimsy"],
    "suspenseful":   ["suspense", "suspenseful", "mystery", "unknown",
                      "secret", "hidden", "clue", "investigation",
                      "thriller", "whodunit", "detective"],
    "romantic":      ["romance", "romantic", "love", "passion",
                      "relationship", "desire", "affair", "attraction",
                      "longing", "infatuation"],
    "adventurous":   ["adventure", "adventurous", "journey", "quest",
                      "explore", "discovery", "expedition", "travel",
                      "mission", "odyssey"],
    "eerie":         ["eerie", "haunting", "ghost", "supernatural",
                      "uncanny", "unsettling", "creepy", "horror",
                      "chilling", "dread", "foreboding"],
    "bittersweet":   ["bittersweet", "mixed", "ambiguous", "complicated",
                      "beauty and sadness", "joy and sorrow",
                      "happiness and loss", "complex emotion"],
    "inspiring":     ["inspiring", "inspirational", "courage", "strength",
                      "determination", "persevere", "hero", "brave",
                      "resilience", "remarkable", "extraordinary"],
    "humorous":      ["funny", "comedy", "humor", "humour", "laugh",
                      "wit", "satire", "absurd", "comic", "ridiculous",
                      "hilarious", "joke"],
    "gritty":        ["gritty", "raw", "realistic", "harsh", "street",
                      "underground", "criminal", "poverty", "unfiltered",
                      "no-holds-barred", "unflinching"],
}


def get_mood_tags_local(aggregated_text: str, top_n: int = 5) -> list[str]:
    """
    Assign mood tags using keyword frequency scoring.
    """
    if not aggregated_text or not aggregated_text.strip():
        return []
    text_lower = aggregated_text.lower()
    scores: dict[str, int] = {}
    for mood, keywords in _MOOD_KEYWORDS.items():
        count = sum(text_lower.count(kw) for kw in keywords)
        if count > 0:
            scores[mood] = count
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [mood for mood, _ in ranked[:top_n]]


# ── Index builder ─────────────────────────────────────────────────────────────

def build_and_save_index(
    movies_df: pd.DataFrame,
    books_df: pd.DataFrame,
    movie_title_col: str = "movie_name",
    book_title_col: str = "title",
    movie_genre_col: str = "movie_genres",
    book_genre_col: str = "assigned_genre",
    rating_col: str = "rating",
) -> None:
    """
    Groups reviews by unique title, concatenates plot summaries with up to 15
    user reviews, extracts genre tags, then builds and persists embeddings and
    metadata for both the movie and book domains.

    Outputs (written to INDEX_DIR):
        movie_embeddings.npy / movie_meta.pkl
        book_embeddings.npy  / book_meta.pkl
    """
    model = SentenceTransformer(MODEL_NAME)
    # Mood tagging is now always local — parameter kept for back-compat
    print("  [mood] Local keyword mood-tagging enabled (no API required).")

    domain_configs = [
        (movies_df, movie_title_col, movie_genre_col, "movie"),
        (books_df,  book_title_col,  book_genre_col,  "book"),
    ]

    for df, title_col, genre_col, label in domain_configs:
        print(f"\n[{label}] Aggregating reviews per title …")
        df = df.copy().reset_index(drop=True)

        # Ensure clean_title exists
        if "clean_title" not in df.columns:
            df["clean_title"] = df[title_col].apply(_clean_title)

        rows: list[dict] = []
        print(f"  Processing {df[title_col].nunique():,} unique titles …")

        for title, group in df.groupby(title_col):
            clean = group["clean_title"].iloc[0]
            review_count = len(group)

            # 1. Plot summary (first non-empty)
            plot_text = ""
            if "plot_summary" in group.columns:
                plots = group["plot_summary"].dropna()
                if not plots.empty:
                    plot_text = str(plots.iloc[0]).strip()

            # 2. Up to 15 distinct review texts
            review_texts: list[str] = []
            if "review_detail" in group.columns:
                for rev in group["review_detail"].dropna().head(15):
                    rev_str = str(rev).strip()
                    if rev_str and rev_str.lower() not in ("nan", "none", ""):
                        review_texts.append(rev_str)

            # 3. Structured aggregated text
            parts = []
            if plot_text:
                parts.append(f"Plot: {plot_text}")
            if review_texts:
                parts.append(f"Reviews and Vibe: {' '.join(review_texts)}")
            aggregated_text = "\n".join(parts)[:2000]

            # 4. Sentiment from average rating
            if rating_col in group.columns:
                avg_rating = pd.to_numeric(group[rating_col], errors="coerce").mean()
                sentiment = bool(avg_rating >= SENTIMENT_THRESHOLD) if pd.notna(avg_rating) else True
            else:
                sentiment = True

            # 5. Normalised genre list
            genres_list: list[str] = []
            if genre_col and genre_col in group.columns:
                non_empty = group[genre_col].dropna()
                if not non_empty.empty:
                    genres_list = _parse_genres(non_empty.iloc[0])

            # 6. Mood tags (optional — requires Gemini)
            mood_tags: list[str] = []
            mood_tags = get_mood_tags_local(aggregated_text)

            rows.append({
                "title":           title,
                "clean_title":     clean,
                "aggregated_text": aggregated_text,
                "review_count":    review_count,
                "domain":          label,
                "sentiment":       sentiment,
                "genres":          genres_list,
                "mood_tags":       mood_tags,
            })

        meta = pd.DataFrame(rows)

        print(f"  Embedding {len(meta):,} titles …")
        emb = model.encode(
            meta["aggregated_text"].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        np.save(INDEX_DIR / f"{label}_embeddings.npy", emb)
        meta.to_pickle(INDEX_DIR / f"{label}_meta.pkl")
        print(f"  ✓ Saved {label} index  ({emb.nbytes / 1e6:.1f} MB)")


df_movies = pd.read_csv(r"D:\Holberton_Final\data\imdb_cmu_final_with_plots.csv.gz", compression="gzip")
df_books = pd.read_csv(r"D:\Holberton_Final\data\goodreads_multigenre_master.csv")
build_and_save_index(movies_df=df_movies, books_df=df_books)
