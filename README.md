# Plagiarism / Text Similarity Checker

NLP Lab Mini-Project by **Rachit** and **Manan**. This is the backup to our Smart Review Analyzer project.

Upload two documents, or a whole class's submissions. The app scores how similar they are using 8 methods: word n-grams, winnowing, LCS, TF-IDF, POS content words, an n-gram language model and sentence embeddings. It then combines the scores with a trained model and highlights every sentence as **exact copy**, **light edit**, **paraphrase** or **original**.

## How it covers the lab syllabus

| Lab experiment | Where it is in this project |
|---|---|
| 1. CSV + graphs | `src/data.py`, `src/plots.py`: ROC curves, boxplots, similarity heatmap |
| 2. Tokenization | `src/preprocessing.py`; sentences drive the alignment, words drive the n-grams |
| 3. Preprocessing, morphology | stopwords, lemmatization, the POS and morphology table in the app |
| 4. Word count, NER | shared phrases, shared named entities (`src/linguistics.py`) |
| 5. POS tagging, chunking | content-word (POS) similarity, shared noun phrases |
| 6. N-gram LM + smoothing | **the core:** n-gram Jaccard and containment, winnowing, and an add-k smoothed LM (`src/ngram_lm.py`) |
| 7. Sentiment | a warning when matched sentences flip sentiment ("effective" vs "not effective") |
| 8. Summarization | a summary of the copied content |

## Setup (once, on each laptop)

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
```bash
pip install -r requirements.txt
```
```bash
python -m spacy download en_core_web_sm
```
```bash
python scripts/download_data.py
```

The download script fetches the Clough corpus, MRPC, PAWS and the MiniLM model (about 90 MB).

## Run

```bash
streamlit run app.py
```
```bash
python -m pytest -q
```

## Experiments

Results go to `results/` and figures to `results/figures/`.

| Command | What it does |
|---|---|
| `python scripts/experiments.py clough` | Document-level test on 95 answers with 5-fold cross-validation. Trains the combined model used by the app. |
| `python scripts/experiments.py mrpc` | Sentence-level paraphrase test: thresholds tuned on MRPC train, tested on MRPC test |
| `python scripts/experiments.py paws` | Stress test where both sentences share almost every word but can mean different things |
| `python scripts/experiments.py table` | One F1 table across all three datasets |

Each method is treated as a classifier: a pair counts as plagiarised if its score is at or above a threshold. The threshold is always chosen on training data and never on the test data.

## Results so far (F1)

| Method | Clough (documents) | MRPC (sentences) | PAWS (word-swap trap) |
|---|---|---|---|
| Jaccard (1-gram) | 0.955 | 0.808 | 0.628 |
| Containment (3-gram) | 0.955 | 0.799 | 0.679 |
| LCS | 0.907 | 0.797 | 0.685 |
| TF-IDF cosine | 0.964 | 0.798 | 0.662 |
| Content words (POS) | 0.964 | 0.802 | 0.624 |
| Semantic (MiniLM) | 0.883 | 0.812 | 0.621 |
| **Combined (logistic regression)** | **0.964** | **0.818** | **0.718** |
| Majority baseline | 0.750 | 0.799 | 0.000 |

Talking points for the report and viva:

- **Clough:** word overlap is enough for copy-style plagiarism. Embeddings do worse because every answer is on the same topic, so even honest answers look semantically similar (average 0.57).
- **MRPC:** single methods barely beat the majority baseline (66.5% accuracy). The combined model reaches 74.0%.
- **PAWS:** bag-of-words methods are at chance (ROC-AUC about 0.50) because they ignore word order. Order-aware methods do better (3-gram, LCS: AUC 0.75–0.78). The combined model scores 76% accuracy.

## Datasets

- **Clough & Stevenson (2011):** 95 short answers about 5 computer-science topics. Each is labelled cut & paste, light revision, heavy revision or non-plagiarised. The original Sheffield site is offline, so we use the copy bundled with the Pattern library (CLiPS).
- **MRPC:** about 5.8k sentence pairs labelled paraphrase or not. About 67% are paraphrases, so always compare against the majority baseline.
- **PAWS:** 108k pairs with nearly identical wording. Word-overlap methods fail on it.
- **`data/samples/`:** our own demo texts: a source, a lightly edited copy, a paraphrase and an unrelated text.

## Project structure

```
app.py                three modes: two documents, many documents (heatmap), how it works
src/
  similarity.py       Jaccard, containment, winnowing, LCS, TF-IDF, n-gram LM score
  embeddings.py       MiniLM sentence embeddings, semantic coverage
  linguistics.py      spaCy: POS content words, entities, noun chunks
  alignment.py        sentence matching + labels, sentiment flip, summary
  scoring.py          all scores for many pairs + the combined logistic-regression model
  evaluation.py       threshold tuning, metrics, baseline
  data.py             Clough / MRPC / PAWS loaders, reading .txt/.pdf/.docx
  preprocessing.py, ngram_lm.py   shared with the review analyzer
scripts/              download_data.py, experiments.py
tests/test_core.py
```

## Who does what

| You (Rachit): methods and evaluation | Manan: app and linguistics |
|---|---|
| `similarity.py`, `ngram_lm.py`, `evaluation.py`, `scripts/experiments.py`, `results/` | `app.py`, `alignment.py`, `linguistics.py`, `plots.py`, demo texts |

## Known limitations

- **Embeddings and negation:** embeddings can rate "X is safe" and "X is not safe" as very similar. The sentiment-flip warning only partly helps.
- **Cross-language plagiarism:** not detected. A multilingual model, `paraphrase-multilingual-MiniLM-L12-v2`, could add it as future work.
- **AI-generated text:** not detected, on purpose. Such detectors are unreliable; OpenAI withdrew its own detector in 2023.
- **Clough is small:** 95 pairs, so the results come with wide error bars.
