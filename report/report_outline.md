# Plagiarism / Text Similarity Checker: Report Outline

Aim for about 15–25 pages. Copy the tables and figures from `results/`.

## Abstract
- The problem
- The 8 methods
- The datasets
- The best F1 on Clough and on MRPC
- One key finding, e.g. "lexical methods fail on paraphrase; embeddings fix that but fail on PAWS"

## 1. Introduction
- Academic integrity
- Why copy-paste detection is easy and paraphrase detection is hard
- Objectives
- What we contribute: the same evaluation across 8 methods and 3 datasets, plus a tool that highlights matches sentence by sentence

## 2. Related work
- **Fingerprinting:** MOSS and winnowing (Schleimer, Wilkerson & Aiken 2003); the guarantee that any shared run of w + k − 1 words is detected
- **N-gram overlap:** Clough & Stevenson 2011; the PAN plagiarism-detection competitions
- **Paraphrase detection:** MRPC (Dolan & Brockett 2005); PAWS (Zhang et al. 2019)
- **Sentence-BERT:** Reimers & Gurevych 2019 (SBERT gets 76.0 accuracy on MRPC, vs 71.1 for BERT's [CLS] embedding)
- **Existing tools:** Turnitin, JPlag, Dolos, and simple TF-IDF scripts; what they lack
- **AI-text detectors:** why we don't build one (OpenAI's classifier caught only 26% of AI-written text and was withdrawn)

## 3. Datasets
- A table with each dataset's size, labels and source
- Clough's four plagiarism levels; how we turned them into plagiarised vs not
- MRPC's class imbalance (about 67% paraphrases) and the majority baseline

## 4. Methodology
- **4.1 Architecture diagram:** input → preprocessing → 8 scorers → combined model → sentence alignment → UI
- **4.2 Preprocessing (Exp 2, 3):** which methods keep stopwords (n-grams, LCS) and which remove them (TF-IDF)
- **4.3 Lexical methods (Exp 6):** formulas for Jaccard, containment, winnowing, the LCS dynamic program, and TF-IDF cosine
- **4.4 N-gram LM:** add-k smoothing; perplexity of the suspicious text under the source's LM
- **4.5 POS and NER (Exp 4, 5):** content-word Jaccard; shared entities and noun phrases
- **4.6 Semantic similarity:** MiniLM; why we embed sentences rather than whole documents (the 256-token limit); semantic coverage
- **4.7 Combined model:** logistic regression on all the scores
- **4.8 Sentence labels:** exact copy / light edit / paraphrase / original thresholds; the sentiment flip (Exp 7); the summary (Exp 8)
- **4.9 Evaluation protocol:** thresholds tuned on training data only; 5-fold cross-validation for Clough

## 5. Results
- **Table 1:** Clough results, `results/clough_results.csv`
- **Figure:** `clough_boxplots.png`, showing scores fall as obfuscation rises (cut > light > heavy > non)
- **Table 2:** MRPC results with the majority baseline, `results/mrpc_results.csv`
- **Table 3:** PAWS stress test, `results/paws_results.csv`; explain why every lexical method collapses on it
- **Figure:** `f1_by_dataset.png` and the ROC curves
- **Error analysis:** 5–10 false positives and false negatives, with an explanation for each

## 6. Application demo
- A screenshot of each mode, plus the heatmap on one Clough topic

## 7. Limitations and future work
- Cross-language plagiarism, code plagiarism, a larger corpus (PAN-PC-11), fine-tuning

## 8. Conclusion

## References
- Clough & Stevenson 2011
- Schleimer et al. 2003
- Dolan & Brockett 2005
- Zhang et al. 2019 (PAWS)
- Reimers & Gurevych 2019
- Jurafsky & Martin, Chapter 3
- The spaCy, NLTK, scikit-learn and sentence-transformers docs
