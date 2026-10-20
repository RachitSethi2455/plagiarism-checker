"""N-gram language model with smoothing, written from scratch (Lab Exp 6).
(Shared with our Smart Review Analyzer project.)

Used in this project as a plagiarism signal: train an LM on the SOURCE
document, then measure the perplexity of the SUSPICIOUS document. Copied text
reuses the source's word sequences, so the LM is "less surprised" by it
(low perplexity); independently written text gets high perplexity.
(PerplexityClassifier below is kept from the review project for reference.)

Smoothing: add-k (k=1 is Laplace).
    P(w | h) = (count(h, w) + k) / (count(h) + k * V)
Without smoothing, any unseen n-gram gets probability 0 and perplexity becomes infinite.
"""
import math
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class NGramLM:
    def __init__(self, n=2, k=1.0, min_count=2):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n, self.k, self.min_count = n, k, min_count
        self.vocab = set()
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.next_words = defaultdict(Counter)

    def _pad(self, tokens):
        tokens = [t if t in self.vocab else UNK for t in tokens]
        return [BOS] * (self.n - 1) + tokens + [EOS]

    @staticmethod
    def build_vocab(token_lists, min_count=2):
        """Words seen at least min_count times; everything else becomes <unk>."""
        word_counts = Counter(t for toks in token_lists for t in toks)
        return {w for w, c in word_counts.items() if c >= min_count} | {UNK, EOS}

    def fit(self, token_lists, vocab=None):
        self.vocab = vocab or self.build_vocab(token_lists, self.min_count)
        for toks in token_lists:
            padded = self._pad(toks)
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1:i])
                word = padded[i]
                self.ngram_counts[context + (word,)] += 1
                self.context_counts[context] += 1
                self.next_words[context][word] += 1
        return self

    @property
    def V(self):
        return len(self.vocab)

    def prob(self, word, context=()):
        """Smoothed P(word | last n-1 words of context)."""
        context = tuple(t if t in self.vocab or t == BOS else UNK for t in context)
        context = ((BOS,) * (self.n - 1) + context)[-(self.n - 1):] if self.n > 1 else ()
        word = word if word in self.vocab else UNK
        return (self.ngram_counts[context + (word,)] + self.k) / (self.context_counts[context] + self.k * self.V)

    def log_prob(self, tokens):
        padded = self._pad(tokens)
        total = 0.0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1:i])
            word = padded[i]
            p = (self.ngram_counts[context + (word,)] + self.k) / (self.context_counts[context] + self.k * self.V)
            total += math.log(p)
        return total, len(padded) - (self.n - 1)

    def perplexity(self, tokens):
        """PP = exp(-(1/N) * sum log P(w_i | history)). Lower = better fit."""
        log_p, count = self.log_prob(tokens)
        return math.exp(-log_p / max(count, 1))

    def suggest(self, context, top=5):
        """Most likely next words after the given context (list of tokens)."""
        context = tuple(t if t in self.vocab else UNK for t in context)
        context = ((BOS,) * (self.n - 1) + context)[-(self.n - 1):] if self.n > 1 else ()
        candidates = self.next_words.get(context, Counter())
        return [(w, self.prob(w, context)) for w, _ in candidates.most_common(top + 2)
                if w not in (UNK, EOS)][:top]


class PerplexityClassifier:
    """One language model per class; predict the class with the lowest perplexity."""

    def __init__(self, n=2, k=1.0, min_count=2):
        self.params = dict(n=n, k=k, min_count=min_count)
        self.models = {}

    def fit(self, token_lists, labels):
        # One shared vocabulary: otherwise a word seen only in negative reviews would be <unk>
        # in the positive model, and the two models would not be comparable.
        vocab = NGramLM.build_vocab(token_lists, self.params["min_count"])
        by_label = defaultdict(list)
        for toks, label in zip(token_lists, labels):
            by_label[label].append(toks)
        self.models = {label: NGramLM(**self.params).fit(docs, vocab) for label, docs in by_label.items()}
        return self

    def predict_one(self, tokens):
        return min(self.models, key=lambda label: self.models[label].perplexity(tokens))

    def predict(self, token_lists):
        return [self.predict_one(toks) for toks in token_lists]
