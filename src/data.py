"""Datasets and file reading (Lab Exp 1: loading data with pandas)."""
import io
import urllib.request

import pandas as pd

from src.config import CLOUGH_CSV, CLOUGH_URL

CLOUGH_LEVELS = {0: "non-plagiarised", 1: "heavy revision", 2: "light revision", 3: "cut & paste"}
TOPIC_KEYWORDS = {"dynamic programming": "Dynamic programming", "pagerank": "PageRank",
                  "inheritance": "Inheritance (OOP)", "vector space": "Vector space model",
                  "bayes": "Bayes' theorem"}


def download_clough():
    """Clough & Stevenson (2011) corpus. The original Sheffield host is down, so we use the
    copy bundled in the Pattern library's test data (CLiPS, BSD licence)."""
    urllib.request.urlretrieve(CLOUGH_URL, CLOUGH_CSV)


def _topic(source_text):
    low = source_text.lower()
    return next((name for key, name in TOPIC_KEYWORDS.items() if key in low[:400]), "Other")


def load_clough():
    """95 student answers, each with its Wikipedia source and plagiarism level."""
    if not CLOUGH_CSV.exists():
        download_clough()
    df = pd.read_csv(CLOUGH_CSV, header=None, names=["answer", "source", "level"], encoding="utf-8-sig")
    df["level_name"] = df["level"].map(CLOUGH_LEVELS)
    df["plagiarised"] = (df["level"] > 0).astype(int)
    df["topic"] = df["source"].map(_topic)
    return df


def load_pairs(name, split):
    """Sentence-pair paraphrase datasets from Hugging Face -> DataFrame(text_a, text_b, label)."""
    from datasets import load_dataset
    if name == "mrpc":
        ds = load_dataset("nyu-mll/glue", "mrpc", split=split)
    elif name == "paws":
        ds = load_dataset("google-research-datasets/paws", "labeled_final", split=split)
    else:
        raise ValueError(name)
    df = ds.to_pandas()
    return df.rename(columns={"sentence1": "text_a", "sentence2": "text_b"})[["text_a", "text_b", "label"]]


def read_uploaded(file):
    """Text from an uploaded .txt / .pdf / .docx (Streamlit UploadedFile or path)."""
    name = getattr(file, "name", str(file)).lower()
    data = file.read() if hasattr(file, "read") else open(file, "rb").read()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if name.endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="replace")
