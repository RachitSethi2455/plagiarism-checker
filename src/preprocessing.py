"""Text cleaning, tokenization and preprocessing (Lab Exp 2 & 3).

Negation words ("not", "no", "never", ...) are kept by default: NLTK's stopword
list removes them, which turns "not good" into "good" and hurts sentiment.
"""
import html
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

NLTK_PACKAGES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}

NEGATIONS = {"not", "no", "nor", "never", "none", "nothing", "nobody", "neither", "cannot", "without"}

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_SPACES = re.compile(r"\s+")


def ensure_nltk_data():
    """Download the NLTK resources we need (only the first time)."""
    for package, path in NLTK_PACKAGES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_data()
_lemmatizer = WordNetLemmatizer()
_stemmer = PorterStemmer()


def clean_text(text):
    """Remove HTML (IMDB has lots of <br />), URLs and extra whitespace."""
    text = html.unescape(str(text))
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def sentences(text):
    """Sentence tokenization."""
    return sent_tokenize(clean_text(text))


def tokens(text):
    """Word tokenization (keeps punctuation as separate tokens)."""
    return word_tokenize(clean_text(text))


@lru_cache(maxsize=2)
def get_stopwords(keep_negations=True):
    words = set(stopwords.words("english"))
    if keep_negations:
        # word_tokenize splits "isn't" into "is" + "n't", and preprocess() maps "n't" -> "not"
        words = {w for w in words if w not in NEGATIONS and not w.endswith("n't")}
    return frozenset(words)


@lru_cache(maxsize=100_000)
def lemmatize(word):
    # WordNet lemmatizer defaults to nouns; try verb form too ("loved" -> "love")
    noun = _lemmatizer.lemmatize(word)
    return _lemmatizer.lemmatize(noun, pos="v")


def stem(word):
    return _stemmer.stem(word)


def preprocess(text, remove_stopwords=True, use_lemmatize=True, keep_negations=True):
    """Full pipeline: clean -> lowercase -> tokenize -> filter -> lemmatize.

    Returns a list of tokens.
    """
    stops = get_stopwords(keep_negations) if remove_stopwords else frozenset()
    out = []
    for tok in word_tokenize(clean_text(text).lower()):
        if tok == "n't":
            tok = "not"
        if not tok.isalpha() or tok in stops:
            continue
        out.append(lemmatize(tok) if use_lemmatize else tok)
    return out


def preprocess_to_string(text, **kwargs):
    """Same as preprocess() but joined back into one string (for TF-IDF)."""
    return " ".join(preprocess(text, **kwargs))


def preprocessing_steps(text):
    """Show every step on one text - handy for the app and the viva."""
    cleaned = clean_text(text)
    toks = word_tokenize(cleaned.lower())
    no_punct = [t for t in toks if t.isalpha()]
    stops = get_stopwords(True)
    no_stop = [t for t in no_punct if t not in stops]
    return {
        "cleaned": cleaned,
        "sentences": sent_tokenize(cleaned),
        "tokens": toks,
        "without_punctuation": no_punct,
        "without_stopwords": no_stop,
        "stemmed": [stem(t) for t in no_stop],
        "lemmatized": [lemmatize(t) for t in no_stop],
    }
