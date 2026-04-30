from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.stroke_app.preprocessing import StrokeSchema, clean_dataframe, split_X_y
from src.stroke_app.training import TrainConfig, load_model, save_model, train_and_evaluate
from src.stroke_app.plots import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_numeric_histograms,
    plot_pr,
    plot_roc,
    plot_target_distribution,
)


APP_TITLE = "Stroke Prediction – ML Model Explorer"
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "stroke_model.joblib"
DATA_PATH = Path("data") / "healthcare-dataset-stroke-data.csv"


def _ensure_artifacts_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _schema() -> StrokeSchema:
    return StrokeSchema()


def _sidebar_config() -> tuple[str, TrainConfig]:
    st.sidebar.header("Training")

    model_name = st.sidebar.selectbox(
        "Model",
        options=["Random Forest", "SVM", "Decision Tree", "KNN", "Naive Bayes"],
        index=0,
    )

    use_smote = st.sidebar.toggle("Use SMOTE", value=True)
    smote_k = st.sidebar.slider("SMOTE k_neighbors", min_value=3, max_value=10, value=5)

    test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    cfg = TrainConfig(test_size=float(test_size), use_smote=bool(use_smote), smote_k_neighbors=int(smote_k))
    return model_name, cfg


def _predict_form(model) -> None:
    schema = _schema()
    st.subheader("Single-patient prediction")

    with st.form("predict"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("gender", options=["Male", "Female", "Other"], index=0)
            age = st.number_input("age", min_value=0.0, max_value=120.0, value=50.0, step=1.0)
            hypertension = st.selectbox("hypertension", options=[0, 1], index=0)
            heart_disease = st.selectbox("heart_disease", options=[0, 1], index=0)
            ever_married = st.selectbox("ever_married", options=["Yes", "No"], index=0)

        with col2:
            work_type = st.selectbox(
                "work_type",
                options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                index=0,
            )
            residence_type = st.selectbox("Residence_type", options=["Urban", "Rural"], index=0)
            avg_glucose_level = st.number_input(
                "avg_glucose_level", min_value=0.0, max_value=400.0, value=110.0, step=1.0
            )
            bmi = st.number_input("bmi", min_value=0.0, max_value=100.0, value=28.0, step=0.5)
            smoking_status = st.selectbox(
                "smoking_status",
                options=["never smoked", "formerly smoked", "smokes", "Unknown"],
                index=0,
            )

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    row = pd.DataFrame(
        [
            {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": residence_type,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "smoking_status": smoking_status,
            }
        ]
    )

    pred = int(model.predict(row)[0])

    prob: Optional[float] = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(row)[0, 1])

    st.markdown("**Result**")
    if prob is None:
        st.write({"predicted_class": pred})
    else:
        st.write({"predicted_class": pred, "stroke_probability": round(prob, 4)})


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _ensure_artifacts_dir()

    st.title(APP_TITLE)
    st.caption("Upload your dataset, train a model, and visualize its performance.")

    model_name, train_cfg = _sidebar_config()

    if not DATA_PATH.exists():
        st.error(
            "Bundled dataset not found. Expected file at: "
            f"{DATA_PATH}. Make sure it exists in the repo."
        )
        st.stop()

    st.info(f"Using bundled dataset: {DATA_PATH}")

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    df: Optional[pd.DataFrame] = None
    try:
        df = clean_dataframe(load_csv_from_path(str(DATA_PATH)))
    except Exception as e:
        st.error(f"Failed to load bundled dataset: {e}")
        st.stop()

    with col_left:
        st.subheader("Data")
        if df is None:
            st.info("Upload a CSV to begin.")
        else:
            st.write("Shape:", df.shape)
            st.dataframe(df.head(20), use_container_width=True)

            if "stroke" in df.columns:
                st.pyplot(plot_target_distribution(df, target_col="stroke"))

            schema = _schema()
            st.pyplot(plot_numeric_histograms(df, numeric_cols=list(schema.numeric_features)))
            st.pyplot(plot_correlation_heatmap(df))

    with col_right:
        st.subheader("Model")

        if st.button("Train / Retrain", type="primary", disabled=(df is None)):
            if df is None:
                st.stop()

            schema = _schema()
            X, y = split_X_y(df, schema)

            with st.spinner("Training model..."):
                model, metrics = train_and_evaluate(
                    X=X,
                    y=y,
                    model_name=model_name,
                    schema=schema,
                    config=train_cfg,
                )

            st.session_state["model"] = model
            try:
                save_model(model, str(MODEL_PATH))
                st.success(f"Trained and saved model to {MODEL_PATH}")
            except Exception as e:
                st.warning(
                    "Model trained but could not be saved to disk. "
                    f"Using in-session model only. Error: {e}"
                )
            st.session_state["metrics"] = metrics

        model = st.session_state.get("model")
        if model is None and MODEL_PATH.exists():
            try:
                model = load_model(str(MODEL_PATH))
                st.session_state["model"] = model
                st.write("Loaded model:", str(MODEL_PATH))
            except Exception as e:
                st.warning(f"Failed to load saved model: {e}")

        metrics: Optional[Dict[str, Any]] = st.session_state.get("metrics")
        if metrics:
            st.markdown("**Metrics**")
            display = {
                k: v
                for k, v in metrics.items()
                if k
                in {
                    "model",
                    "n_train",
                    "n_test",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                }
            }
            st.json(display)

            st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"]))

            roc_fig = plot_roc(metrics)
            if roc_fig is not None:
                st.pyplot(roc_fig)

            pr_fig = plot_pr(metrics)
            if pr_fig is not None:
                st.pyplot(pr_fig)

        st.divider()

        if model is None:
            st.warning("No trained model loaded yet. Train first.")
        else:
            _predict_form(model)


if __name__ == "__main__":
    main()
