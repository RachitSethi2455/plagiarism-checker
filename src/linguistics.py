"""spaCy-based signals: content-word overlap via POS (Exp 5), shared named entities (Exp 4),
noun-chunk overlap (Exp 5 chunking) and morphology (Exp 3)."""
from functools import lru_cache

import pandas as pd
import spacy

from src.config import SPACY_MODEL

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ"}


@lru_cache(maxsize=1)
def get_nlp():
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise OSError(f"spaCy model missing. Run: python -m spacy download {SPACY_MODEL}") from exc


def parse(text):
    return get_nlp()(text)


def parse_many(texts, batch_size=32):
    return list(get_nlp().pipe(texts, batch_size=batch_size))


def content_lemmas(doc):
    """Only nouns, verbs, adjectives: ignores function words that every text shares."""
    return {t.lemma_.lower() for t in doc if t.pos_ in CONTENT_POS and t.is_alpha and not t.is_stop}


def content_word_jaccard(sus_doc, src_doc):
    a, b = content_lemmas(sus_doc), content_lemmas(src_doc)
    return len(a & b) / len(a | b) if a or b else 0.0


def entities(doc):
    return {(e.text.strip(), e.label_) for e in doc.ents}


def shared_entities(sus_doc, src_doc):
    shared = entities(sus_doc) & entities(src_doc)
    return pd.DataFrame(sorted(shared), columns=["entity", "label"])


def noun_chunk_overlap(sus_doc, src_doc):
    a = {c.lemma_.lower() for c in sus_doc.noun_chunks}
    b = {c.lemma_.lower() for c in src_doc.noun_chunks}
    return sorted(a & b)


def pos_table(doc):
    return pd.DataFrame([{"word": t.text, "lemma": t.lemma_, "pos": t.pos_, "morph": str(t.morph) or "-",
                          "content_word": t.pos_ in CONTENT_POS and not t.is_stop}
                         for t in doc if not (t.is_space or t.is_punct)])
