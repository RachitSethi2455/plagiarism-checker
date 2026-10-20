"""All experiments for the report. Results -> results/*.csv, figures -> results/figures/.

Usage (from the project folder):
    python scripts/experiments.py clough      # document-level plagiarism (95 answers, 5-fold CV)
    python scripts/experiments.py mrpc        # sentence-level paraphrase (MRPC train -> test)
    python scripts/experiments.py paws        # stress test: same words, different meaning
    python scripts/experiments.py table       # one F1 table across all datasets

Add --no-embeddings to skip MiniLM, --sample N to use fewer pairs (mrpc/paws).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import FIGURES_DIR, RANDOM_STATE, RESULTS_DIR
from src.data import CLOUGH_LEVELS, load_clough, load_pairs
from src.evaluation import best_threshold, majority_baseline, metrics, save
from src.plots import boxplot_by_level, roc_figure
from src.scoring import build_combined, save_combined, score_pairs

COMBINED = "Combined (logistic regression)"


def timed_scores(sus, src, args):
    start = time.time()
    df = score_pairs(sus, src, use_embeddings=not args.no_embeddings)
    print(f"Scored {len(df)} pairs with {df.shape[1]} methods in {time.time() - start:.0f}s")
    return df


# ---------------------------------------------------------------- Clough (document level)
def cmd_clough(args):
    data = load_clough()
    print(data["level_name"].value_counts().to_string())
    scores = timed_scores(data["answer"].tolist(), data["source"].tolist(), args)
    pd.concat([data[["topic", "level", "level_name", "plagiarised"]], scores], axis=1).to_csv(
        RESULTS_DIR / "clough_scores.csv", index=False)
    y = data["plagiarised"].to_numpy()

    # 95 pairs is small, so use stratified 5-fold cross-validation:
    # tune each threshold on 4 folds, test on the 5th, repeat.
    folds = list(StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(scores, y))
    rows = []
    for method in scores.columns:
        s = scores[method].to_numpy()
        preds, thresholds = np.zeros_like(y), []
        for train, test in folds:
            t = best_threshold(s[train], y[train])
            thresholds.append(t)
            preds[test] = s[test] >= t
        rows.append({"method": method, "threshold_mean": round(np.mean(thresholds), 3), **metrics(y, preds, s)})

    combined = build_combined()
    proba = cross_val_predict(combined, scores, y, cv=folds, method="predict_proba")[:, 1]
    rows.append({"method": COMBINED, "threshold_mean": 0.5, **metrics(y, (proba >= 0.5).astype(int), proba)})
    rows.append({"method": "Majority baseline (always 1)", **metrics(y, np.ones_like(y))})
    save(rows, "clough_results.csv")

    # final combined model for the app: trained on all 95 pairs
    save_combined(build_combined().fit(scores, y), scores.columns)

    by_level = pd.concat([data["level_name"], scores], axis=1).groupby("level_name").mean().round(3)
    by_level = by_level.reindex([CLOUGH_LEVELS[i] for i in (3, 2, 1, 0)])
    by_level.to_csv(RESULTS_DIR / "clough_mean_by_level.csv")
    print("\nMean score per plagiarism level:\n", by_level.T.to_string())

    order = [CLOUGH_LEVELS[i] for i in (3, 2, 1, 0)]
    plot_df = pd.concat([data["level_name"], scores], axis=1)
    boxplot_by_level(plot_df, list(scores.columns), "level_name", order,
                     "Clough corpus: scores by plagiarism level").savefig(FIGURES_DIR / "clough_boxplots.png", dpi=150)
    roc_table = scores.assign(**{COMBINED: proba})
    roc_figure(y, roc_table, "Clough corpus: ROC curves (5-fold CV)").savefig(FIGURES_DIR / "clough_roc.png", dpi=150)
    print(f"Figures saved to {FIGURES_DIR}")


# ---------------------------------------------------------------- MRPC / PAWS (sentence level)
def run_pairs(name, train_split, test_split, args, default_sample):
    n = args.sample or default_sample
    train = load_pairs(name, train_split)
    test = load_pairs(name, test_split)
    if n:
        train = train.sample(min(n, len(train)), random_state=RANDOM_STATE)
        test = test.sample(min(n // 2, len(test)), random_state=RANDOM_STATE)
    print(f"{name}: {len(train)} train pairs, {len(test)} test pairs, "
          f"{test['label'].mean():.0%} of test pairs are paraphrases")
    s_train = timed_scores(train["text_a"].tolist(), train["text_b"].tolist(), args)
    s_test = timed_scores(test["text_a"].tolist(), test["text_b"].tolist(), args)
    y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()

    rows = []
    for method in s_train.columns:
        t = best_threshold(s_train[method], y_train)
        rows.append({"method": method, "threshold": round(t, 3),
                     **metrics(y_test, (s_test[method] >= t).astype(int), s_test[method])})
    combined = build_combined().fit(s_train, y_train)
    proba = combined.predict_proba(s_test)[:, 1]
    rows.append({"method": COMBINED, "threshold": 0.5, **metrics(y_test, (proba >= 0.5).astype(int), proba)})
    rows.append(majority_baseline(y_train, y_test))
    save(rows, f"{name}_results.csv")
    roc_figure(y_test, s_test.assign(**{COMBINED: proba}), f"{name.upper()}: ROC curves (test set)").savefig(
        FIGURES_DIR / f"{name}_roc.png", dpi=150)


def cmd_mrpc(args):
    run_pairs("mrpc", "train", "test", args, default_sample=None)


def cmd_paws(args):
    run_pairs("paws", "train", "test", args, default_sample=4000)


# ---------------------------------------------------------------- summary table
def cmd_table(args):
    import matplotlib.pyplot as plt

    frames = []
    for name in ("clough", "mrpc", "paws"):
        path = RESULTS_DIR / f"{name}_results.csv"
        if path.exists():
            frames.append(pd.read_csv(path).set_index("method")["f1"].rename(name))
    if not frames:
        sys.exit("No results yet - run the other commands first.")
    table = pd.concat(frames, axis=1)
    table.to_csv(RESULTS_DIR / "f1_by_dataset.csv")
    print(table.to_string())

    ax = table.drop(index=[i for i in table.index if i.startswith("Majority")]).plot.barh(figsize=(8, 6))
    ax.set_xlabel("F1 (plagiarised / paraphrase class)")
    ax.set_title("F1 of each method on each dataset")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "f1_by_dataset.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'f1_by_dataset.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["clough", "mrpc", "paws", "table"])
    parser.add_argument("--sample", type=int, help="number of training pairs (test = half)")
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()
    globals()[f"cmd_{args.command}"](args)


if __name__ == "__main__":
    main()
