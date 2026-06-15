import os
from pathlib import Path
from dotenv import load_dotenv

# ── Environment & paths ───────────────────────────────────────────────────────

load_dotenv()

# Resolve the repo root as 3 levels up from this file:
#   src/matcher/config.py  →  src/matcher  →  src  →  repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

INDEX_DIR = Path(os.getenv("INDEX_DIR", _REPO_ROOT / "index"))
INDEX_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", _REPO_ROOT / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
BATCH_SIZE = 64

# Ratings are on a 0-10 scale; ≥ 6.0 is treated as "positive sentiment".
SENTIMENT_THRESHOLD = 6.0

# Gemini model used for explanation generation
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

# ── Genre normalisation ───────────────────────────────────────────────────────

GENRE_NORMALIZATION_MAP: dict[str, str] = {
    # → Mystery & Crime
    "crime fiction":    "mystery & crime",
    "crime drama":      "mystery & crime",
    "thriller":         "mystery & crime",
    "mystery":          "mystery & crime",
    "film noir":        "mystery & crime",
    "black-and-white":  "mystery & crime",
    # → Fantasy & Paranormal
    "science fiction":  "fantasy & paranormal",
    "fantasy":          "fantasy & paranormal",
    "superhero movie":  "fantasy & paranormal",
    "animation":        "fantasy & paranormal",
    "supernatural":     "fantasy & paranormal",
    "horror":           "fantasy & paranormal",
    # → Romance
    "romance":          "romance",
    "romantic comedy":  "romance",
    "family drama":     "romance",
    "drama":            "romance",
    # → Young Adult
    "adventure":        "young adult",
    "action/adventure": "young adult",
    "period piece":     "young adult",
    "coming of age":    "young adult",
}
