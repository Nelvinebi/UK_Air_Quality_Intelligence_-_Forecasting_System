import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.inference import predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_pm25.pkl"
IMPUTER_PATH = PROJECT_ROOT / "models" / "imputer.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
FEATURE_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "london_air_quality_features_2021_2024.csv"
)


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_required_model_artifacts_exist():
    assert MODEL_PATH.exists(), f"Missing model artifact: {MODEL_PATH}"
    assert IMPUTER_PATH.exists(), f"Missing imputer artifact: {IMPUTER_PATH}"
    assert METADATA_PATH.exists(), f"Missing model metadata: {METADATA_PATH}"


def test_model_and_imputer_load_successfully():
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    assert hasattr(model, "predict")
    assert hasattr(imputer, "transform")


def test_metadata_contains_required_fields():
    metadata = load_metadata()

    assert "features" in metadata
    assert "target" in metadata
    assert "model_params" in metadata

    assert isinstance(metadata["features"], list)
    assert len(metadata["features"]) > 0
    assert isinstance(metadata["target"], str)
    assert isinstance(metadata["model_params"], dict)


def test_metadata_feature_count_matches_model():
    metadata = load_metadata()
    model = joblib.load(MODEL_PATH)

    features = metadata["features"]

    assert model.n_features_in_ == len(features)


def test_processed_feature_dataset_exists():
    assert FEATURE_DATA_PATH.exists(), f"Missing processed feature dataset: {FEATURE_DATA_PATH}"


def test_all_model_features_exist_in_processed_dataset():
    metadata = load_metadata()

    df = pd.read_csv(FEATURE_DATA_PATH, nrows=10)

    missing_features = [feature for feature in metadata["features"] if feature not in df.columns]

    assert missing_features == []


def test_target_exists_in_processed_dataset():
    metadata = load_metadata()

    df = pd.read_csv(FEATURE_DATA_PATH, nrows=10)

    assert metadata["target"] in df.columns


def test_imputer_output_feature_count_matches_model():
    metadata = load_metadata()

    df = pd.read_csv(FEATURE_DATA_PATH)

    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    features = metadata["features"]

    sample = df[features].iloc[[100]].copy()

    transformed = imputer.transform(sample)

    assert transformed.shape == (1, len(features))
    assert transformed.shape[1] == model.n_features_in_


def test_saved_model_can_generate_prediction():
    metadata = load_metadata()

    df = pd.read_csv(FEATURE_DATA_PATH)

    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    prediction = predict(
        model=model,
        imputer=imputer,
        data=df.iloc[[100]],
        features=metadata["features"],
    )

    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])


def test_prediction_is_non_negative_for_sample_row():
    metadata = load_metadata()

    df = pd.read_csv(FEATURE_DATA_PATH)

    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    prediction = predict(
        model=model,
        imputer=imputer,
        data=df.iloc[[100]],
        features=metadata["features"],
    )

    assert prediction[0] >= 0


def test_sample_target_is_available():
    metadata = load_metadata()

    df = pd.read_csv(FEATURE_DATA_PATH)

    target = metadata["target"]

    value = df.iloc[100][target]

    assert pd.notna(value)
    assert np.isfinite(float(value))
