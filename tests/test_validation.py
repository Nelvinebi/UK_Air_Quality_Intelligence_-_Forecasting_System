import json

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from src.validation import (
    load_model_metadata,
    validate_file_exists,
    validate_model_artifacts,
    validate_model_dataset,
    validate_required_columns,
    validate_station_datetime_integrity,
)


def make_valid_dataframe():
    return pd.DataFrame(
        {
            "Station": [
                "Station A",
                "Station A",
            ],
            "Datetime": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                ]
            ),
            "PM2.5": [10.0, 11.0],
            "Temperature": [5.0, 6.0],
            "PM2.5_target": [11.0, 12.0],
        }
    )


def test_validate_file_exists_raises_for_missing_file(tmp_path):
    missing = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Required file not found",
    ):
        validate_file_exists(missing)


def test_validate_required_columns_detects_missing_column():
    df = pd.DataFrame(
        {"PM2.5": [10.0]}
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_required_columns(
            df,
            ["PM2.5", "Temperature"],
        )


def test_station_datetime_validation_accepts_valid_data():
    df = make_valid_dataframe()

    validate_station_datetime_integrity(df)


def test_station_datetime_validation_detects_duplicates():
    df = make_valid_dataframe()

    df = pd.concat(
        [df, df.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate Station-Datetime",
    ):
        validate_station_datetime_integrity(df)


def test_station_datetime_validation_detects_invalid_datetime():
    df = make_valid_dataframe()

    df["Datetime"] = [
        "2024-01-01 00:00:00",
        "not-a-date",
    ]

    with pytest.raises(
        ValueError,
        match="Invalid Datetime",
    ):
        validate_station_datetime_integrity(df)


def test_model_dataset_detects_missing_target():
    df = make_valid_dataframe()

    df.loc[0, "PM2.5_target"] = None

    with pytest.raises(
        ValueError,
        match="Missing target values",
    ):
        validate_model_dataset(
            df,
            features=[
                "PM2.5",
                "Temperature",
            ],
            target="PM2.5_target",
        )


def test_load_model_metadata_requires_environment(tmp_path):
    path = tmp_path / "metadata.json"

    metadata = {
        "features": ["PM2.5"],
        "target": "PM2.5_target",
        "model_params": {},
    }

    path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Missing model metadata keys",
    ):
        load_model_metadata(path)


def test_validate_model_artifacts_accepts_matching_bundle(
    tmp_path,
):
    X = pd.DataFrame(
        {
            "PM2.5": [10.0, 20.0, 30.0],
            "Temperature": [5.0, 6.0, 7.0],
        }
    )

    y = pd.Series(
        [11.0, 21.0, 31.0]
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imputed = imputer.fit_transform(X)

    X_imputed = pd.DataFrame(
        X_imputed,
        columns=X.columns,
    )

    model = RandomForestRegressor(
        n_estimators=2,
        random_state=42,
    )

    model.fit(
        X_imputed,
        y,
    )

    model_path = (
        tmp_path / "model.pkl"
    )

    imputer_path = (
        tmp_path / "imputer.pkl"
    )

    metadata_path = (
        tmp_path / "metadata.json"
    )

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        imputer,
        imputer_path,
    )

    metadata = {
        "features": [
            "PM2.5",
            "Temperature",
        ],
        "target": "PM2.5_target",
        "model_params": {},
        "environment": {
            "python": "3.13",
            "scikit_learn": "1.8.0",
            "joblib": "1.5.3",
        },
    }

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    result = validate_model_artifacts(
        model_path=model_path,
        imputer_path=imputer_path,
        metadata_path=metadata_path,
    )

    assert result["features"] == [
        "PM2.5",
        "Temperature",
    ]
