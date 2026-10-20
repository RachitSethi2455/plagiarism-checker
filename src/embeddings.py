"""Semantic similarity with sentence embeddings - catches paraphrases that share few words.

Model: all-MiniLM-L6-v2 (22.7M parameters, runs on a laptop CPU).
It truncates input after 256 word pieces, so we embed SENTENCES, not whole documents.

Document score = "semantic coverage": for each sentence of the suspicious doc,
take its best cosine match among source sentences, then average.
"""
from functools import lru_cache

import numpy as np

from src.config import EMBEDDING_MODEL
from src.preprocessing import sentences


@lru_cache(maxsize=2)
def get_model(name=EMBEDDING_MODEL):
    """Returns the SentenceTransformer, or None if sentence-transformers is not installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer(name, device="cpu")


def embed(texts, model_name=EMBEDDING_MODEL, batch_size=64):
    """Unit-length vectors, so a dot product = cosine similarity."""
    model = get_model(model_name)
    if model is None:
        return None
    return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=len(texts) > 500)


def similarity_matrix(sus_sentences, src_sentences, model_name=EMBEDDING_MODEL):
    vecs = embed(list(sus_sentences) + list(src_sentences), model_name)
    if vecs is None:
        return None
    a, b = vecs[:len(sus_sentences)], vecs[len(sus_sentences):]
    return a @ b.T


def semantic_coverage(sus_text, src_text, model_name=EMBEDDING_MODEL):
    sus_s, src_s = sentences(sus_text), sentences(src_text)
    if not sus_s or not src_s:
        return 0.0
    sim = similarity_matrix(sus_s, src_s, model_name)
    return None if sim is None else float(sim.max(axis=1).mean())


def semantic_coverage_batch(pairs, model_name=EMBEDDING_MODEL):
    """Same as semantic_coverage for many (sus, src) pairs, embedding each unique sentence once."""
    split = [(sentences(a), sentences(b)) for a, b in pairs]
    unique = sorted({s for a, b in split for s in a + b})
    vecs = embed(unique, model_name)
    if vecs is None:
        return None
    index = {s: i for i, s in enumerate(unique)}
    scores = []
    for a, b in split:
        if not a or not b:
            scores.append(0.0)
            continue
        sim = vecs[[index[s] for s in a]] @ vecs[[index[s] for s in b]].T
        scores.append(float(sim.max(axis=1).mean()))
    return np.array(scores)
