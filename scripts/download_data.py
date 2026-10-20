"""Download the datasets.

    python scripts/download_data.py

- Clough & Stevenson plagiarism corpus (95 answers + 5 sources), from the copy
  bundled with the Pattern library (the original Sheffield site is offline).
- MRPC and PAWS paraphrase datasets from Hugging Face (cached automatically).
- The MiniLM embedding model (~90 MB, cached automatically).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embeddings
from src.config import CLOUGH_CSV
from src.data import download_clough, load_clough, load_pairs


def main():
    download_clough()
    df = load_clough()
    print(f"Clough: {len(df)} answers -> {CLOUGH_CSV}")
    print(df["level_name"].value_counts().to_string(), "\n")
    for name, splits in (("mrpc", ("train", "test")), ("paws", ("train", "test"))):
        for split in splits:
            print(f"{name} {split}: {len(load_pairs(name, split)):,} pairs")
    if embeddings.get_model() is not None:
        print("\nMiniLM embedding model ready.")


if __name__ == "__main__":
    main()
