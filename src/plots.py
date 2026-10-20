"""Charts (Lab Exp 1 & 4)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def roc_figure(labels, score_table, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in score_table.columns:
        fpr, tpr, _ = roc_curve(labels, score_table[name])
        ax.plot(fpr, tpr, label=name, lw=1.5)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def boxplot_by_level(df, methods, level_col, order, title):
    cols = 3
    rows = int(np.ceil(len(methods) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, m in zip(axes.flat, methods):
        ax.boxplot([df.loc[df[level_col] == lvl, m] for lvl in order], showfliers=False)
        ax.set_xticks(range(1, len(order) + 1), [o.replace(" ", "\n") for o in order], fontsize=7)
        ax.set_title(m, fontsize=9)
    for ax in list(axes.flat)[len(methods):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def heatmap_figure(matrix, labels, title):
    fig, ax = plt.subplots(figsize=(0.6 * len(labels) + 3, 0.6 * len(labels) + 2))
    im = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if matrix[i][j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def method_bar_figure(scores, title="Similarity by method"):
    names = list(scores.keys())
    vals = [scores[n] for n in names]
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(names) + 1))
    ax.barh(names[::-1], vals[::-1], color="#c62828")
    ax.set_xlim(0, 1)
    for i, v in enumerate(vals[::-1]):
        ax.text(min(v + 0.01, 0.9), i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
