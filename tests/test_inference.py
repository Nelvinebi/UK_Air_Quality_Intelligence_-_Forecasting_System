import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.inference import (
    predict,
    predict_row,
    prepare_features,
    validate_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_pm25.pkl"
IMPUTER_PATH = PROJECT_ROOT / "models" / "imputer.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "london_air_quality_features_2021_2024.csv"
)


def load_bundle():
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, imputer, metadata


def load_sample():
    return pd.read_csv(DATA_PATH).iloc[:5].copy()


def test_validate_features_returns_ordered_feature_list():
    _, _, metadata = load_bundle()

    sample = load_sample()

    features = validate_features(
        sample,
        metadata["features"],
    )

    assert features == metadata["features"]


def test_validate_features_raises_for_missing_feature():
    df = pd.DataFrame({"PM2.5": [10.0]})

    with pytest.raises(
        ValueError,
        match="Missing required model features",
    ):
        validate_features(
            df,
            ["PM2.5", "Temperature"],
        )


def test_prepare_features_preserves_feature_names():
    _, imputer, metadata = load_bundle()

    sample = load_sample()

    prepared = prepare_features(
        data=sample,
        imputer=imputer,
        features=metadata["features"],
    )

    assert isinstance(prepared, pd.DataFrame)
    assert list(prepared.columns) == metadata["features"]


def test_prepare_features_preserves_row_count():
    _, imputer, metadata = load_bundle()

    sample = load_sample()

    prepared = prepare_features(
        data=sample,
        imputer=imputer,
        features=metadata["features"],
    )

    assert len(prepared) == len(sample)


def test_prepare_features_contains_no_missing_values():
    _, imputer, metadata = load_bundle()

    sample = load_sample()

    prepared = prepare_features(
        data=sample,
        imputer=imputer,
        features=metadata["features"],
    )

    assert not prepared.isna().any().any()


def test_predict_returns_one_prediction_per_row():
    model, imputer, metadata = load_bundle()

    sample = load_sample()

    predictions = predict(
        model=model,
        imputer=imputer,
        data=sample,
        features=metadata["features"],
    )

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(sample),)


def test_predict_returns_finite_values():
    model, imputer, metadata = load_bundle()

    sample = load_sample()

    predictions = predict(
        model=model,
        imputer=imputer,
        data=sample,
        features=metadata["features"],
    )

    assert np.isfinite(predictions).all()


def test_predict_row_returns_float():
    model, imputer, metadata = load_bundle()

    sample = load_sample().iloc[0]

    prediction = predict_row(
        model=model,
        imputer=imputer,
        row=sample,
        features=metadata["features"],
    )

    assert isinstance(prediction, float)
    assert np.isfinite(prediction)
