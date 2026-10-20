"""Quick checks. Run: python -m pytest -q"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import alignment, embeddings, similarity
from src.config import SAMPLES_DIR
from src.evaluation import best_threshold
from src.similarity import Document

SOURCE = (SAMPLES_DIR / "source.txt").read_text(encoding="utf-8")
COPIED = (SAMPLES_DIR / "copied.txt").read_text(encoding="utf-8")
PARA = (SAMPLES_DIR / "paraphrased.txt").read_text(encoding="utf-8")
UNRELATED = (SAMPLES_DIR / "original.txt").read_text(encoding="utf-8")


def test_jaccard_and_containment():
    a, b = {1, 2, 3, 4}, {3, 4, 5, 6, 7, 8}
    assert similarity.jaccard(a, b) == 2 / 8
    assert similarity.containment(a, b) == 2 / 4


def test_identical_documents_score_one():
    d = Document(SOURCE)
    assert similarity.ngram_jaccard(d, d, 3) == 1.0
    assert similarity.winnowing_similarity(d, d) == 1.0
    assert similarity.lcs_similarity(d, d) == 1.0
    assert math.isclose(similarity.tfidf_cosine(d, d), 1.0)


def test_lcs_length():
    assert similarity.lcs_length("a b c d e".split(), "a x c y e".split()) == 3
    assert similarity.lcs_length([], ["a"]) == 0


def test_winnowing_guarantee():
    """A shared run of w + k - 1 = 8 words must give at least one shared fingerprint."""
    shared = "the quick brown fox jumps over the lazy".split()
    a = "one two three".split() + shared + "four five".split()
    b = "alpha beta".split() + shared + "gamma delta epsilon".split()
    assert similarity.fingerprints(a, 5, 4) & similarity.fingerprints(b, 5, 4)


def test_copy_scores_higher_than_unrelated():
    src = Document(SOURCE)
    copied, unrelated = Document(COPIED), Document(UNRELATED)
    for fn in (lambda s: similarity.ngram_containment(s, src), lambda s: similarity.lcs_similarity(s, src),
               lambda s: similarity.lm_score(s, src)):
        assert fn(copied) > fn(unrelated)


def test_best_threshold():
    assert best_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 0.8


def test_semantic_catches_paraphrase():
    if embeddings.get_model() is None:
        return  # sentence-transformers not installed
    para = embeddings.semantic_coverage(PARA, SOURCE)
    unrelated = embeddings.semantic_coverage(UNRELATED, SOURCE)
    assert para > 0.6 > unrelated
    # ...while word overlap barely notices the paraphrase
    assert similarity.ngram_containment(Document(PARA), Document(SOURCE)) < 0.1


def test_alignment_labels():
    matches, _, _ = alignment.align(COPIED, SOURCE, use_embeddings=False)
    labels = [m["label"] for m in matches]
    assert labels[0] == "exact copy"
    assert labels[-1] == "original"


def test_sentiment_flip():
    assert alignment.sentiment_flip("The treatment is very effective and safe.",
                                    "The treatment is not effective and is unsafe.")
