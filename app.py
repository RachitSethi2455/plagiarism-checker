"""Plagiarism / Text Similarity Checker - Streamlit app.

Run:  streamlit run app.py
"""
import html
import itertools
import math

import numpy as np
import pandas as pd
import streamlit as st

from src import alignment, embeddings, linguistics, scoring, similarity
from src.config import CLOUGH_CSV, SAMPLES_DIR
from src.data import load_clough, read_uploaded
from src.plots import heatmap_figure, method_bar_figure
from src.preprocessing import preprocessing_steps

st.set_page_config(page_title="Plagiarism Checker", page_icon="🔍", layout="wide")

EXAMPLES = {
    "Copied with light edits": "copied.txt",
    "Paraphrased (same meaning, different words)": "paraphrased.txt",
    "Unrelated text": "original.txt",
}
LEGEND = " ".join(f"<span style='background:{c}33;border-bottom:3px solid {c};padding:1px 6px'>{l}</span>"
                  for l, c in alignment.LABEL_COLORS.items() if c)


# ---------------------------------------------------------------- cached work
@st.cache_resource(show_spinner="Loading models (first run downloads ~90 MB)...")
def warm_up():
    linguistics.get_nlp()
    return embeddings.get_model() is not None


@st.cache_data(show_spinner="Comparing documents...")
def analyse(sus_text, src_text, use_emb):
    scores = scoring.score_pairs([sus_text], [src_text], use_embeddings=use_emb).iloc[0].to_dict()
    matches, sus_s, src_s = alignment.align(sus_text, src_text, use_embeddings=use_emb)
    return scores, matches, sus_s, src_s


@st.cache_data(show_spinner=False)
def clough():
    return load_clough() if CLOUGH_CSV.exists() else None


def read_sample(name):
    return (SAMPLES_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------- rendering helpers
def render_sentences(sents, labels, tooltips):
    parts = []
    for s, label, tip in zip(sents, labels, tooltips):
        color = alignment.LABEL_COLORS.get(label)
        safe = html.escape(s)
        if color:
            parts.append(f"<span title='{html.escape(tip)}' style='background:{color}33;"
                         f"border-bottom:3px solid {color};padding:1px 2px'>{safe}</span>")
        else:
            parts.append(safe)
    return "<div style='line-height:1.9'>" + " ".join(parts) + "</div>"


def verdict(prob):
    if prob >= 0.7:
        return "🔴 Likely plagiarised"
    if prob >= 0.4:
        return "🟠 Possibly plagiarised - review the highlighted parts"
    return "🟢 Likely original"


def text_input(column, title, key, default):
    column.subheader(title)
    upload = column.file_uploader("Upload .txt / .pdf / .docx", type=["txt", "pdf", "docx"], key=f"up_{key}")
    if upload is not None:
        default = read_uploaded(upload)
    return column.text_area("Text", default, height=260, key=f"text_{key}_{hash(default)}")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🔍 Plagiarism Checker")
st.sidebar.caption("NLP Lab Mini-Project")
mode = st.sidebar.radio("Mode", ["Compare two documents", "Compare many documents", "How it works"])
has_embeddings = warm_up()
use_emb = st.sidebar.checkbox("Semantic matching (MiniLM)", value=has_embeddings, disabled=not has_embeddings,
                              help="Detects paraphrases. Needs sentence-transformers.")
bundle = scoring.load_combined()
if bundle is None:
    st.sidebar.warning("Combined model not trained yet - run `python scripts/experiments.py clough`.")

# ================================================================ MODE 1: two documents
if mode == "Compare two documents":
    st.title("Compare two documents")
    options = list(EXAMPLES) + (["Clough corpus answer"] if clough() is not None else [])
    example = st.selectbox("Load an example", options)
    if example == "Clough corpus answer":
        df = clough()
        level = st.selectbox("Plagiarism level", ["cut & paste", "light revision", "heavy revision", "non-plagiarised"])
        subset = df[df["level_name"] == level].reset_index(drop=True)
        idx = st.slider("Answer #", 0, len(subset) - 1, 0)
        sus_default, src_default = subset.loc[idx, "answer"], subset.loc[idx, "source"]
        st.caption(f"Topic: {subset.loc[idx, 'topic']} — ground truth: {level}")
    else:
        sus_default, src_default = read_sample(EXAMPLES[example]), read_sample("source.txt")

    left, right = st.columns(2)
    sus_text = text_input(left, "📄 Suspicious document", "sus", sus_default)
    src_text = text_input(right, "📚 Source document", "src", src_default)
    if not sus_text.strip() or not src_text.strip():
        st.info("Enter both documents.")
        st.stop()

    scores, matches, sus_s, src_s = analyse(sus_text, src_text, use_emb)
    prob = scoring.combined_probability(scores, bundle) if bundle else None
    share = alignment.match_summary(matches)

    st.divider()
    c = st.columns(5)
    matched = 1 - share["original"] / 100
    # The combined model was trained on Clough (copy-style plagiarism), so it under-weights
    # paraphrase; the sentence-level matches catch that case.
    risk = matched if prob is None else max(prob, matched)
    c[0].metric("Plagiarism risk", f"{risk:.0%}",
                help="Higher of: the combined model's probability"
                     + (f" ({prob:.0%})" if prob is not None else " (not trained yet)")
                     + f" and the share of sentences matched to the source ({matched:.0%}).")
    c[0].markdown(f"**{verdict(risk)}**")
    c[1].metric("Exact copy", f"{share['exact copy']}%", help="share of suspicious sentences")
    c[2].metric("Light edit", f"{share['light edit']}%")
    c[3].metric("Paraphrase", f"{share['paraphrase']}%")
    c[4].metric("Original", f"{share['original']}%")

    tabs = st.tabs(["🖍️ Highlighted matches", "📊 Scores by method", "🔗 Shared phrases & entities",
                    "📝 Summary of copied content", "⚙️ Preprocessing & POS"])

    with tabs[0]:
        st.markdown(LEGEND + " &nbsp; Hover over a sentence to see its match.", unsafe_allow_html=True)
        src_labels = ["original"] * len(src_s)
        rank = {"exact copy": 3, "light edit": 2, "paraphrase": 1, "original": 0}
        for m in matches:
            j = m["src_index"]
            if j is not None and rank[m["label"]] > rank[src_labels[j]]:
                src_labels[j] = m["label"]
        sus_tips = [f"{m['label']} | lexical {m['lexical']} | semantic {m['semantic']} | source: {m['src_sentence']}"
                    for m in matches]
        a, b = st.columns(2)
        a.markdown("**Suspicious document**")
        a.markdown(render_sentences(sus_s, [m["label"] for m in matches], sus_tips), unsafe_allow_html=True)
        b.markdown("**Source document**")
        b.markdown(render_sentences(src_s, src_labels, [""] * len(src_s)), unsafe_allow_html=True)

        flips = [m for m in matches if m["sentiment_flip"]]
        if flips:
            st.warning(f"⚠️ {len(flips)} matched sentence(s) flip sentiment (e.g. 'effective' → 'not effective'). "
                       "They look similar but may mean the opposite.")
        with st.expander("Sentence-by-sentence table"):
            st.dataframe(pd.DataFrame(matches).drop(columns=["sus_index", "src_index"]),
                         hide_index=True, use_container_width=True)

    with tabs[1]:
        bounded = {k: v for k, v in scores.items() if not k.startswith("N-gram LM")}
        l, r = st.columns([1.3, 1])
        l.pyplot(method_bar_figure(bounded))
        pp = math.exp(-scores["N-gram LM (-log PP)"])
        r.metric("N-gram LM perplexity", f"{pp:,.0f}",
                 help="Bigram LM with add-k smoothing trained on the source. Low perplexity = the suspicious "
                      "text reuses the source's word sequences.")
        r.markdown("""
| Method | Catches | Misses |
|---|---|---|
| Jaccard / Containment | copy-paste | paraphrase |
| Winnowing | copied passages | paraphrase |
| LCS | word order kept | reordering |
| TF-IDF cosine | same topic words | synonyms |
| Content words (POS) | key nouns/verbs | synonyms |
| Semantic (MiniLM) | paraphrase | negation traps |
""")

    with tabs[2]:
        sus_doc, src_doc = similarity.Document(sus_text), similarity.Document(src_text)
        l, r = st.columns(2)
        l.markdown("**Longest shared phrases (word 5-grams)**")
        phrases = similarity.shared_ngrams(sus_doc, src_doc, n=5, top=20)
        l.write("\n".join(f"- {p}" for p in phrases) or "None")
        sp, dp = linguistics.parse(sus_text), linguistics.parse(src_text)
        r.markdown("**Shared named entities**")
        ents = linguistics.shared_entities(sp, dp)
        if ents.empty:
            r.write("None")
        else:
            r.dataframe(ents, hide_index=True)
        r.markdown("**Shared noun phrases**")
        r.write(", ".join(linguistics.noun_chunk_overlap(sp, dp)) or "None")

    with tabs[3]:
        copied = alignment.copied_text(matches)
        if copied:
            st.markdown("**What was taken from the source, in brief:**")
            st.write(alignment.frequency_summary(copied, 3))
        else:
            st.write("No matched content to summarise.")

    with tabs[4]:
        steps = preprocessing_steps(sus_text)
        for name in ("sentences", "tokens", "without_stopwords", "lemmatized"):
            st.markdown(f"**{name.replace('_', ' ').title()}**")
            st.write(steps[name] if name == "sentences" else " | ".join(steps[name]))
        st.markdown("**POS tags and morphology** (content words are used by the POS method)")
        st.dataframe(linguistics.pos_table(linguistics.parse(sus_text)), hide_index=True, use_container_width=True)

# ================================================================ MODE 2: many documents
elif mode == "Compare many documents":
    st.title("Compare many documents")
    st.caption("Upload a whole class's submissions to find pairs that are suspiciously similar.")
    files = st.file_uploader("Upload 3+ files (.txt / .pdf / .docx)", type=["txt", "pdf", "docx"],
                             accept_multiple_files=True)
    docs = {f.name: read_uploaded(f) for f in files} if files else {}
    if not docs:
        df = clough()
        if df is not None:
            topic = st.selectbox("…or use Clough answers for one topic", sorted(df["topic"].unique()))
            sub = df[df["topic"] == topic].reset_index(drop=True)
            docs = {f"{i}: {r.level_name}": r.answer for i, r in sub.iterrows()}
            docs["SOURCE (Wikipedia)"] = sub.loc[0, "source"]
        else:
            docs = {n: read_sample(f) for n, f in
                    [("source", "source.txt"), ("copied", "copied.txt"), ("paraphrased", "paraphrased.txt"),
                     ("unrelated", "original.txt")]}
    methods = ["TF-IDF cosine", "Jaccard (3-gram)"] + (["Semantic (MiniLM)"] if use_emb else [])
    method = st.selectbox("Method", methods)

    names, texts = list(docs), list(docs.values())
    with st.spinner(f"Comparing {len(names)} documents..."):
        n = len(names)
        matrix = np.eye(n)
        if method == "Semantic (MiniLM)":
            pairs = list(itertools.combinations(range(n), 2))
            vals = embeddings.semantic_coverage_batch([(texts[i], texts[j]) for i, j in pairs])
            back = embeddings.semantic_coverage_batch([(texts[j], texts[i]) for i, j in pairs])
            for (i, j), v, w in zip(pairs, vals, back):
                matrix[i, j] = matrix[j, i] = (v + w) / 2
        else:
            d = [similarity.Document(t) for t in texts]
            vec = similarity.fit_tfidf(d)
            for i, j in itertools.combinations(range(n), 2):
                v = (similarity.tfidf_cosine(d[i], d[j], vec) if method == "TF-IDF cosine"
                     else similarity.ngram_jaccard(d[i], d[j], 3))
                matrix[i, j] = matrix[j, i] = v
    st.pyplot(heatmap_figure(matrix, [x[:28] for x in names], f"Pairwise similarity ({method})"))
    top = sorted(((matrix[i, j], names[i], names[j]) for i, j in itertools.combinations(range(n), 2)), reverse=True)
    st.subheader("Most similar pairs")
    st.dataframe(pd.DataFrame(top[:15], columns=["similarity", "document A", "document B"]).round(3),
                 hide_index=True, use_container_width=True)

# ================================================================ MODE 3: how it works
else:
    st.title("How it works")
    st.markdown("""
**Pipeline:** clean text → sentence + word tokenization → (stopword removal, lemmatization) →
7 lexical scores + 1 semantic score → combined logistic-regression model → sentence-level highlighting.

| Lab experiment | Where it appears |
|---|---|
| 1. CSV + graphs | Clough/MRPC CSVs, ROC curves, boxplots, heatmap |
| 2. Tokenization | sentence alignment, word n-grams |
| 3. Preprocessing / morphology | stopwords, lemmatization, POS + morphology table |
| 4. Word count / NER | shared phrases, shared named entities |
| 5. POS / chunking | content-word (POS) similarity, shared noun phrases |
| 6. N-gram LM + smoothing | n-gram overlap, winnowing, LM perplexity (add-k smoothing) |
| 7. Sentiment | sentiment-flip warning on matched sentences |
| 8. Summarization | summary of copied content |

**Why several methods?** Copy-paste is easy for word-overlap methods; paraphrase needs embeddings;
embeddings can be fooled by negation ("is" vs "is not"). Combining them is more robust than any single one.
""")
