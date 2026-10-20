"""Treat every similarity score as a classifier: "plagiarised if score >= threshold".

The threshold is ALWAYS tuned on training data and then applied to test data -
picking it on the test set would be cheating.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.config import RESULTS_DIR


def best_threshold(scores, labels):
    """Threshold that maximises F1 on the given (training) data."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    candidates = np.unique(scores)
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        f1 = f1_score(labels, scores >= t, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t)


def metrics(labels, preds, scores=None):
    out = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    if scores is not None and len(set(labels)) > 1:
        out["roc_auc"] = roc_auc_score(labels, scores)
    return {k: round(float(v), 4) for k, v in out.items()}


def majority_baseline(train_labels, test_labels):
    majority = int(round(np.mean(train_labels)))
    preds = np.full(len(test_labels), majority)
    return {"method": f"Majority baseline (always {majority})", **metrics(test_labels, preds)}


def save(rows, filename):
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / filename, index=False)
    print(f"\nSaved {RESULTS_DIR / filename}")
    print(df.to_string(index=False))
    return df
