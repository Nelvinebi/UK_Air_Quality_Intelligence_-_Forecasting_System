"""Cached model, dataset, and report loading for the Streamlit dashboard."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_DIR = PROJECT_ROOT / "models"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "london_air_quality_features_2021_2024.csv"
)

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "baseline_model_results.csv"


@st.cache_resource
def load_model_bundle():
    """Load the trained model, imputer, and persisted model metadata."""
    model = joblib.load(MODEL_DIR / "random_forest_pm25.pkl")
    imputer = joblib.load(MODEL_DIR / "imputer.pkl")

    with open(
        MODEL_DIR / "model_metadata.json",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return model, imputer, metadata


@st.cache_data
def load_test_data():
    """Load the processed dataset and retain held-out 2024 observations."""
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Datetime"],
    )

    return df[df["Datetime"].dt.year == 2024].reset_index(drop=True)


@st.cache_data
def load_reports():
    """Load evaluation reports when their persisted artifacts are available."""
    reports = {}

    for name in [
        "feature_importance.csv",
        "station_level_errors.csv",
    ]:
        path = REPORTS_DIR / name

        reports[name] = (
            pd.read_csv(path)
            if path.exists()
            else None
        )

    reports["baseline_model_results.csv"] = (
        pd.read_csv(RESULTS_PATH)
        if RESULTS_PATH.exists()
        else None
    )

    return reports
