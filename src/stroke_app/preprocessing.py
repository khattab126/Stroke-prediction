from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "stroke"


@dataclass(frozen=True)
class StrokeSchema:
    target: str = TARGET_COL
    numeric_features: Tuple[str, ...] = (
        "age",
        "avg_glucose_level",
        "bmi",
    )
    categorical_features: Tuple[str, ...] = (
        "gender",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    )

    def all_feature_columns(self) -> List[str]:
        return list(self.numeric_features) + list(self.categorical_features)


def _make_ohe() -> OneHotEncoder:
    # sklearn compatibility across versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(schema: StrokeSchema) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", _make_ohe()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(schema.numeric_features)),
            ("cat", categorical_pipe, list(schema.categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Strip whitespace in string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("string").str.strip()

    return df


def split_X_y(df: pd.DataFrame, schema: StrokeSchema) -> Tuple[pd.DataFrame, pd.Series]:
    if schema.target not in df.columns:
        raise ValueError(f"Missing target column '{schema.target}'.")

    missing = [c for c in schema.all_feature_columns() if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required feature columns: " + ", ".join(missing)
        )

    X = df[schema.all_feature_columns()].copy()
    y = df[schema.target].astype(int).copy()
    return X, y
