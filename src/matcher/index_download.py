"""
src/matcher/index_download.py — fetch the prebuilt embedding index from a
Hugging Face *Dataset* repo if it isn't already on disk.

Why this exists
----------------
The index/ folder (movie/book embeddings + metadata) is too large to live
in the Space's git repo (HF Spaces enforce a small repo size limit), so it's
hosted separately as a HF Dataset and pulled down at startup instead.

Setup (one-time)
-----------------
1. Create a Hugging Face *Dataset* repo (Files tab -> Add file -> Upload
   files works the same as for Spaces, but Datasets allow much larger repos)
   and upload exactly these 4 files to its root:
       movie_embeddings.npy
       book_embeddings.npy
       movie_meta.pkl
       book_meta.pkl

2. In your Space's Settings -> Variables and secrets, add a *Variable*
   (not a secret -- it's not sensitive):
       HF_INDEX_REPO = "<your-username>/<your-dataset-name>"

3. If the dataset repo is PRIVATE, also add a *Secret*:
       HF_TOKEN = <a HF access token with read access>
   (Public datasets need no token at all.)

That's it -- ensure_index_files() is called once near the top of app.py,
checks whether the 4 files already exist in INDEX_DIR, and downloads any
that are missing.
"""

from __future__ import annotations

import os
import shutil

from src.matcher.config import INDEX_DIR

REQUIRED_INDEX_FILES = (
    "movie_embeddings.npy",
    "book_embeddings.npy",
    "movie_meta.pkl",
    "book_meta.pkl",
)


def ensure_index_files() -> None:
    """Download any missing index files from HF_INDEX_REPO into INDEX_DIR.

    Safe to call on every Streamlit rerun: if all files are already present
    (local dev, or a previous run already downloaded them), this returns
    immediately without touching the network.
    """
    missing = [f for f in REQUIRED_INDEX_FILES if not (INDEX_DIR / f).exists()]
    if not missing:
        return

    repo_id = os.environ.get("HF_INDEX_REPO")
    if not repo_id:
        raise FileNotFoundError(
            f"Index files {missing} are missing from {INDEX_DIR}, and the "
            "HF_INDEX_REPO environment variable isn't set, so they can't be "
            "downloaded automatically.\n\n"
            "Either put the files in index/ yourself, or set HF_INDEX_REPO "
            "to a Hugging Face dataset repo id, e.g. "
            "'yourname/vibematcher-index' (Space Settings -> Variables)."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download the index from "
            "HF_INDEX_REPO. Add 'huggingface_hub>=0.23.0' to requirements.txt."
        ) from exc

    token = os.environ.get("HF_TOKEN")  # only needed for private datasets

    for fname in missing:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=fname,
            token=token,
        )
        dest = INDEX_DIR / fname
        if not dest.exists():
            shutil.copy(cached_path, dest)
