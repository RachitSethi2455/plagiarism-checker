"""Compute every similarity score for document pairs, and the combined model.

Combined model = logistic regression on all the individual scores, trained on
the Clough corpus. It learns how much to trust each method.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import embeddings, linguistics
from src.config import COMBINED_MODEL_PATH, RANDOM_STATE
from src.similarity import Document, fit_tfidf, lexical_scores

SEMANTIC = "Semantic (MiniLM)"
POS_METHOD = "Content words (POS)"


def score_pairs(sus_texts, src_texts, use_embeddings=True, use_spacy=True, corpus=None):
    """DataFrame with one row per pair and one column per method."""
    sus_docs = [Document(t) for t in sus_texts]
    src_docs = [Document(t) for t in src_texts]
    vectorizer = fit_tfidf(corpus or (sus_docs + src_docs))
    rows = [lexical_scores(s, d, vectorizer) for s, d in zip(sus_docs, src_docs)]
    df = pd.DataFrame(rows)
    if use_spacy:
        sus_p = linguistics.parse_many([d.text for d in sus_docs])
        src_p = linguistics.parse_many([d.text for d in src_docs])
        df[POS_METHOD] = [linguistics.content_word_jaccard(a, b) for a, b in zip(sus_p, src_p)]
    if use_embeddings:
        sem = embeddings.semantic_coverage_batch(list(zip(sus_texts, src_texts)))
        if sem is not None:
            df[SEMANTIC] = sem
    return df


def build_combined():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE))


def save_combined(model, features):
    joblib.dump({"model": model, "features": list(features)}, COMBINED_MODEL_PATH)


def load_combined():
    try:
        return joblib.load(COMBINED_MODEL_PATH)
    except FileNotFoundError:
        return None


def combined_probability(scores_row, bundle=None):
    """Probability of plagiarism from the trained combined model (None if not trained yet)."""
    bundle = bundle or load_combined()
    if bundle is None or any(f not in scores_row for f in bundle["features"]):
        return None
    x = np.array([[scores_row[f] for f in bundle["features"]]])
    return float(bundle["model"].predict_proba(x)[0, 1])
