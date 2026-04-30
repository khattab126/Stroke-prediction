from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump, load
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .preprocessing import StrokeSchema, build_preprocessor


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    use_smote: bool = True
    smote_k_neighbors: int = 5


def make_model(model_name: str, random_state: int = 42) -> BaseEstimator:
    name = model_name.lower().strip()

    if name in {"naive bayes", "nb", "gaussiannb"}:
        return GaussianNB()
    if name in {"svm", "svc"}:
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state)
    if name in {"knn", "k-nearest neighbors", "kneighbors"}:
        return KNeighborsClassifier(n_neighbors=7, weights="distance")
    if name in {"decision tree", "dt", "decisiontree"}:
        return DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=random_state)
    if name in {"random forest", "rf", "randomforest"}:
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model '{model_name}'.")


def build_pipeline(
    model: BaseEstimator,
    schema: Optional[StrokeSchema] = None,
    config: Optional[TrainConfig] = None,
) -> ImbPipeline:
    schema = schema or StrokeSchema()
    config = config or TrainConfig()

    steps = [("preprocess", build_preprocessor(schema))]

    if config.use_smote:
        steps.append(
            (
                "smote",
                SMOTE(random_state=config.random_state, k_neighbors=config.smote_k_neighbors),
            )
        )

    steps.append(("model", model))
    return ImbPipeline(steps=steps)


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    schema: Optional[StrokeSchema] = None,
    config: Optional[TrainConfig] = None,
) -> Tuple[ImbPipeline, Dict[str, Any]]:
    schema = schema or StrokeSchema()
    config = config or TrainConfig()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    model = make_model(model_name=model_name, random_state=config.random_state)
    pipe = build_pipeline(model=model, schema=schema, config=config)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    y_prob: Optional[np.ndarray]
    if hasattr(pipe, "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]
    elif hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X_test)
        # map to [0,1] roughly; keeps ROC/AUC meaningful enough for display
        y_prob = 1 / (1 + np.exp(-scores))
    else:
        y_prob = None

    metrics: Dict[str, Any] = {
        "model": model_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        }

        prec, rec, pr_thresholds = precision_recall_curve(y_test, y_prob)
        metrics["pr_curve"] = {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "thresholds": pr_thresholds.tolist(),
        }

    return pipe, metrics


def save_model(model: ImbPipeline, path: str) -> None:
    dump(model, path)


def load_model(path: str) -> ImbPipeline:
    return load(path)
