"""
build_index.py — One-off script to build / rebuild the VibeMatcher index.

Usage:
    python build_index.py

Environment variables (set in .env):
    MOVIES_CSV   Path to the gzip-compressed movies CSV.
    BOOKS_CSV    Path to the books CSV.
    INDEX_DIR    Destination folder for index files (default: ./index).
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.matcher.index_builder import build_and_save_index

load_dotenv()

MOVIES_CSV = Path(os.getenv("MOVIES_CSV", "data/imdb_cmu_final_with_plots.csv.gz"))
BOOKS_CSV = Path(os.getenv("BOOKS_CSV",  "data/goodreads_multigenre_master.csv"))


def main() -> None:
    print(f"Loading movies from  {MOVIES_CSV} …")
    df_movies = pd.read_csv(MOVIES_CSV, compression="gzip")

    print(f"Loading books from   {BOOKS_CSV} …")
    df_books = pd.read_csv(BOOKS_CSV)

    # The books CSV uses 'movie_name' as the title column — normalise it here
    # so the rest of the pipeline can rely on 'title' consistently.
    if "movie_name" in df_books.columns and "title" not in df_books.columns:
        df_books = df_books.rename(columns={"movie_name": "title"})

    build_and_save_index(
        movies_df=df_movies,
        books_df=df_books,
        movie_title_col="movie_name",
        book_title_col="title",
        movie_genre_col="movie_genres",
        book_genre_col="assigned_genre",
    )
    print("\n✓ Index build complete.")


if __name__ == "__main__":
    main()
