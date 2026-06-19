from huggingface_hub import HfApi

api = HfApi()

print("Starting upload to Hugging Face Spaces...")

# This uploads your whole directory structure smoothly
api.upload_folder(
    folder_path=".",
    repo_id="Gurban01/Holberton_Final",
    repo_type="space",
    ignore_patterns=[
        "*.env",
        "venv/",
        ".git/",
        "__pycache__/",
        "upload.py",       # Don't upload this script itself
        "index/**",        # Heavy embeddings/meta live in a separate HF Dataset
                            # (see upload_index.py + src/matcher/index_download.py)
    ],
)

print("Upload complete! Check your Hugging Face Space.")
