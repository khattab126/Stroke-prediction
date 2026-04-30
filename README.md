# Stroke ML Streamlit App

A simple Streamlit web app to:
- Upload the stroke dataset CSV
- Explore the data (distributions + correlations)
- Train a classification model (optionally with SMOTE)
- Visualize model performance (metrics, confusion matrix, ROC/PR curves)
- Run single-patient predictions

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Dataset requirements

Your CSV must include the target column `stroke` and the following feature columns:

- `gender`
- `age`
- `hypertension`
- `heart_disease`
- `ever_married`
- `work_type`
- `Residence_type`
- `avg_glucose_level`
- `bmi`
- `smoking_status`

If your CSV has an `id` column, it will be ignored.

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repo.
2. In Streamlit Community Cloud, create a new app.
3. Set:
   - **Main file path**: `app.py`
   - **Python requirements**: `requirements.txt`

### Auto-load dataset (no file upload)

If you want the deployed app to automatically load the dataset (instead of uploading a CSV), set `DATASET_URL`.

Option A (recommended): Streamlit Cloud **Secrets**

Add a secret key:

```toml
DATASET_URL = "https://your-hosted-file.csv"
```

Option B: Environment variable

Set `DATASET_URL` to a public CSV URL.

The app trains from the uploaded CSV at runtime and stores a model artifact at `artifacts/stroke_model.joblib`.
