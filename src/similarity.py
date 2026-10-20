"""Lexical similarity methods (Lab Exp 2, 3, 6).

Every function returns a score in [0, 1] where higher = more similar, except
lm_score (a negative log-perplexity, still "higher = more similar").

  Jaccard     |A ∩ B| / |A ∪ B|            symmetric overlap of word n-grams
  Containment |A ∩ B| / |A|                how much of the SUSPICIOUS doc A appears in source B
  Winnowing   containment of fingerprints  MOSS-style document fingerprinting
  LCS         longest common subsequence of words / length of A
  TF-IDF      cosine similarity of TF-IDF vectors
  N-gram LM   -log(perplexity of A under a smoothed LM trained on B)
"""
import math
import zlib
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.ngram_lm import NGramLM
from src.preprocessing import clean_text, preprocess, sentences


@dataclass
class Document:
    """A text plus the token lists every method needs (computed once)."""
    text: str
    tokens: list = field(init=False)    # lowercase words, stopwords KEPT (for n-grams, LCS)
    content: list = field(init=False)   # stopwords removed + lemmatized (for TF-IDF)

    def __post_init__(self):
        self.text = clean_text(self.text)
        self.tokens = preprocess(self.text, remove_stopwords=False, use_lemmatize=False)
        self.content = preprocess(self.text)

    @property
    def sentences(self):
        return sentences(self.text)


# ---------------------------------------------------------------- n-gram overlap
def word_ngrams(tokens, n):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def containment(a, b):
    return len(a & b) / len(a) if a else 0.0


def ngram_jaccard(sus, src, n=3):
    return jaccard(word_ngrams(sus.tokens, n), word_ngrams(src.tokens, n))


def ngram_containment(sus, src, n=3):
    return containment(word_ngrams(sus.tokens, n), word_ngrams(src.tokens, n))


def shared_ngrams(sus, src, n=3, top=15):
    """The longest-running shared phrases are what a teacher wants to see."""
    common = word_ngrams(sus.tokens, n) & word_ngrams(src.tokens, n)
    return [" ".join(g) for g in sorted(common)][:top]


# ---------------------------------------------------------------- winnowing (MOSS)
def _hash(kgram):
    return zlib.crc32(" ".join(kgram).encode("utf-8"))  # deterministic, unlike Python's hash()


def fingerprints(tokens, k=5, w=4):
    """Winnowing (Schleimer et al. 2003): hash every k-gram, then in every window of
    w consecutive hashes keep the minimum. Any shared run of >= w + k - 1 words is
    guaranteed to produce at least one shared fingerprint."""
    hashes = [_hash(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]
    if len(hashes) <= w:
        return set(hashes)
    prints = set()
    for i in range(len(hashes) - w + 1):
        window = hashes[i:i + w]
        prints.add(min(window))
    return prints


def winnowing_similarity(sus, src, k=5, w=4):
    return containment(fingerprints(sus.tokens, k, w), fingerprints(src.tokens, k, w))


# ---------------------------------------------------------------- LCS
def lcs_length(a, b):
    """Dynamic programming, O(len(a) * len(b)) time, O(len(b)) memory."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def lcs_similarity(sus, src):
    return lcs_length(sus.tokens, src.tokens) / len(sus.tokens) if sus.tokens else 0.0


# ---------------------------------------------------------------- TF-IDF
def fit_tfidf(documents):
    """Fit IDF on a whole corpus so common words count less."""
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([" ".join(d.content) for d in documents])


def tfidf_cosine(sus, src, vectorizer=None):
    vectorizer = vectorizer or fit_tfidf([sus, src])
    vecs = vectorizer.transform([" ".join(sus.content), " ".join(src.content)])
    return float(cosine_similarity(vecs[0], vecs[1])[0, 0])


# ---------------------------------------------------------------- N-gram LM (Exp 6)
def lm_perplexity(sus, src, n=2, k=0.01):
    lm = NGramLM(n=n, k=k, min_count=1).fit([src.tokens])
    return lm.perplexity(sus.tokens)


def lm_score(sus, src, n=2, k=0.01):
    """Negative log-perplexity: higher = source LM finds the suspicious text more predictable."""
    return -math.log(lm_perplexity(sus, src, n, k))


# ---------------------------------------------------------------- all lexical methods at once
LEXICAL_METHODS = {
    "Jaccard (1-gram)": lambda s, d, v: ngram_jaccard(s, d, 1),
    "Jaccard (3-gram)": lambda s, d, v: ngram_jaccard(s, d, 3),
    "Containment (3-gram)": lambda s, d, v: ngram_containment(s, d, 3),
    "Winnowing (k=5, w=4)": lambda s, d, v: winnowing_similarity(s, d),
    "LCS": lambda s, d, v: lcs_similarity(s, d),
    "TF-IDF cosine": lambda s, d, v: tfidf_cosine(s, d, v),
    "N-gram LM (-log PP)": lambda s, d, v: lm_score(s, d),
}


def lexical_scores(sus, src, vectorizer=None):
    return {name: fn(sus, src, vectorizer) for name, fn in LEXICAL_METHODS.items()}
