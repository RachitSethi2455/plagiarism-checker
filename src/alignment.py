"""Sentence-level matching between a suspicious document and a source.

Each suspicious sentence gets its best-matching source sentence and a label:
  exact copy   - most word bigrams appear in the source sentence
  light edit   - many bigrams shared (words swapped, a few changed)
  paraphrase   - few words shared, but the embedding says the meaning is the same
  original     - no good match
Also flags matched pairs whose sentiment flips (Exp 7): "is effective" vs "is not effective"
look similar to embeddings but mean the opposite.
"""
from collections import Counter

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src import embeddings
from src.config import EXACT_COPY_MIN, LIGHT_EDIT_MIN, PARAPHRASE_MIN
from src.preprocessing import preprocess, sentences
from src.similarity import containment, word_ngrams

LABEL_COLORS = {"exact copy": "#e53935", "light edit": "#fb8c00", "paraphrase": "#8e24aa", "original": None}
_vader = SentimentIntensityAnalyzer()


def _bigrams(sentence):
    toks = preprocess(sentence, remove_stopwords=False, use_lemmatize=False)
    return word_ngrams(toks, 2) if len(toks) > 1 else word_ngrams(toks, 1)


def label_for(lexical, semantic):
    if lexical >= EXACT_COPY_MIN:
        return "exact copy"
    if lexical >= LIGHT_EDIT_MIN:
        return "light edit"
    if semantic is not None and semantic >= PARAPHRASE_MIN:
        return "paraphrase"
    return "original"


def sentiment_flip(a, b, margin=0.3):
    sa, sb = _vader.polarity_scores(a)["compound"], _vader.polarity_scores(b)["compound"]
    return (sa >= margin and sb <= -margin) or (sa <= -margin and sb >= margin)


def align(sus_text, src_text, use_embeddings=True):
    sus_s, src_s = sentences(sus_text), sentences(src_text)
    if not sus_s or not src_s:
        return [], sus_s, src_s
    sus_g, src_g = [_bigrams(s) for s in sus_s], [_bigrams(s) for s in src_s]
    lexical = np.array([[containment(a, b) for b in src_g] for a in sus_g])
    semantic = embeddings.similarity_matrix(sus_s, src_s) if use_embeddings else None

    matches = []
    for i, s in enumerate(sus_s):
        # best source sentence = highest of (lexical, semantic) evidence
        combined = lexical[i] if semantic is None else np.maximum(lexical[i], semantic[i])
        j = int(combined.argmax())
        lex = float(lexical[i, j])
        sem = None if semantic is None else float(semantic[i, j])
        label = label_for(lex, sem)
        matches.append({
            "sus_index": i, "sus_sentence": s,
            "src_index": j if label != "original" else None,
            "src_sentence": src_s[j] if label != "original" else "",
            "lexical": round(lex, 3), "semantic": None if sem is None else round(sem, 3),
            "label": label,
            "sentiment_flip": label != "original" and sentiment_flip(s, src_s[j]),
        })
    return matches, sus_s, src_s


def match_summary(matches):
    counts = Counter(m["label"] for m in matches)
    total = max(len(matches), 1)
    return {label: round(100 * counts.get(label, 0) / total, 1) for label in LABEL_COLORS}


def copied_text(matches):
    """All suspicious sentences that were matched - input for the summary (Exp 8)."""
    return " ".join(m["sus_sentence"] for m in matches if m["label"] != "original")


def frequency_summary(text, n_sentences=3):
    """Our own extractive summarizer (same idea as Lab Exp 8)."""
    sents = sentences(text)
    if len(sents) <= n_sentences:
        return " ".join(sents)
    freq = Counter(preprocess(text))
    top = max(freq.values()) if freq else 1
    scored = []
    for i, s in enumerate(sents):
        words = preprocess(s)
        scored.append((sum(freq[w] / top for w in words) / (len(words) ** 0.5 or 1), i))
    chosen = sorted(i for _, i in sorted(scored, reverse=True)[:n_sentences])
    return " ".join(sents[i] for i in chosen)
