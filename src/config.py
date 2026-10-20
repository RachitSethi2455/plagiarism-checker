"""Project paths and shared settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SAMPLES_DIR = ROOT / "data" / "samples"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CLOUGH_CSV = DATA_RAW / "clough.csv"
CLOUGH_URL = ("https://raw.githubusercontent.com/clips/pattern/master/test/corpora/"
              "plagiarism-clough%26stevenson.csv")
COMBINED_MODEL_PATH = MODELS_DIR / "combined_clough.joblib"

SPACY_MODEL = "en_core_web_sm"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Sentence-level match labels used for highlighting
EXACT_COPY_MIN = 0.8     # word-bigram containment
LIGHT_EDIT_MIN = 0.4     # word-bigram containment
PARAPHRASE_MIN = 0.6     # embedding cosine; MiniLM's best threshold on MRPC train was 0.604

RANDOM_STATE = 42
