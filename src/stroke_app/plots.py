from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_target_distribution(df: pd.DataFrame, target_col: str = "stroke") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=target_col, ax=ax)
    ax.set_title("Target Distribution")
    ax.set_xlabel(target_col)
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def plot_numeric_histograms(df: pd.DataFrame, numeric_cols: list[str]) -> plt.Figure:
    n = len(numeric_cols)
    fig, axes = plt.subplots(nrows=max(1, n), ncols=1, figsize=(7, 3 * max(1, n)))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        sns.histplot(data=df, x=col, kde=True, ax=ax)
        ax.set_title(f"Distribution: {col}")

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Columns)")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: list[list[int]], labels: Optional[list[str]] = None) -> plt.Figure:
    labels = labels or ["No Stroke", "Stroke"]
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Pred {l}" for l in labels],
        yticklabels=[f"True {l}" for l in labels],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_roc(metrics: Dict[str, Any]) -> Optional[plt.Figure]:
    roc = metrics.get("roc_curve")
    if not roc:
        return None

    fpr = np.asarray(roc["fpr"])
    tpr = np.asarray(roc["tpr"])
    auc = metrics.get("roc_auc")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})" if auc is not None else "ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr(metrics: Dict[str, Any]) -> Optional[plt.Figure]:
    pr = metrics.get("pr_curve")
    if not pr:
        return None

    precision = np.asarray(pr["precision"])
    recall = np.asarray(pr["recall"])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    return fig
