"""
Validation utilities for the London Air Quality Intelligence project.

These checks provide fast, explicit failures for dataset and model-artifact
problems before training, evaluation, inference, or deployment.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import joblib
import pandas as pd


def validate_file_exists(
    path: Path,
    *,
    label: str = "Required file",
) -> Path:
    """Ensure that a required file exists."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")

    return path


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    context: str = "dataset",
) -> list[str]:
    """Ensure that a DataFrame contains all required columns."""
    required = list(required_columns)

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")

    return required


def validate_station_datetime_integrity(
    df: pd.DataFrame,
) -> None:
    """
    Validate the station/time key used throughout the project.

    Checks:
    - Station and Datetime columns exist
    - Datetime values are parseable
    - Station values are present
    - no duplicate (Station, Datetime) pairs exist
    """
    validate_required_columns(
        df,
        ["Station", "Datetime"],
        context="station-time dataset",
    )

    parsed_datetime = pd.to_datetime(
        df["Datetime"],
        errors="coerce",
    )

    invalid_datetime_count = int(parsed_datetime.isna().sum())

    if invalid_datetime_count:
        raise ValueError(f"Invalid Datetime values detected: {invalid_datetime_count}")

    missing_station_count = int(df["Station"].isna().sum())

    if missing_station_count:
        raise ValueError(f"Missing Station values detected: {missing_station_count}")

    duplicate_count = int(df.duplicated(subset=["Station", "Datetime"]).sum())

    if duplicate_count:
        raise ValueError(f"Duplicate Station-Datetime rows detected: {duplicate_count}")


def validate_model_dataset(
    df: pd.DataFrame,
    *,
    features: Iterable[str],
    target: str,
) -> None:
    """
    Validate a model-ready engineered dataset.
    """
    feature_list = list(features)

    validate_required_columns(
        df,
        ["Station", "Datetime", target, *feature_list],
        context="model dataset",
    )

    if df.empty:
        raise ValueError("Model dataset is empty.")

    validate_station_datetime_integrity(df)

    missing_target_count = int(df[target].isna().sum())

    if missing_target_count:
        raise ValueError(f"Missing target values detected: {missing_target_count}")


def load_model_metadata(
    metadata_path: Path,
) -> dict:
    """Load and validate model metadata."""
    validate_file_exists(
        metadata_path,
        label="Model metadata",
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    required_keys = {
        "features",
        "target",
        "model_params",
        "environment",
    }

    missing_keys = sorted(required_keys.difference(metadata))

    if missing_keys:
        raise ValueError(f"Missing model metadata keys: {missing_keys}")

    if not isinstance(metadata["features"], list):
        raise TypeError("Metadata 'features' must be a list.")

    if not metadata["features"]:
        raise ValueError("Metadata feature list is empty.")

    return metadata


def validate_model_artifacts(
    *,
    model_path: Path,
    imputer_path: Path,
    metadata_path: Path,
) -> dict:
    """
    Validate persisted model artifacts and their feature contract.

    Returns the validated metadata dictionary.
    """
    validate_file_exists(
        model_path,
        label="Model artifact",
    )

    validate_file_exists(
        imputer_path,
        label="Imputer artifact",
    )

    metadata = load_model_metadata(metadata_path)

    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)

    if not hasattr(model, "predict"):
        raise ValueError("Model artifact does not provide predict().")

    if not hasattr(imputer, "transform"):
        raise ValueError("Imputer artifact does not provide transform().")

    feature_count = len(metadata["features"])

    model_feature_count = getattr(
        model,
        "n_features_in_",
        None,
    )

    if model_feature_count is not None and model_feature_count != feature_count:
        raise ValueError(
            "Model feature count does not match metadata: "
            f"model={model_feature_count}, "
            f"metadata={feature_count}"
        )

    imputer_feature_count = getattr(
        imputer,
        "n_features_in_",
        None,
    )

    if imputer_feature_count is not None and imputer_feature_count != feature_count:
        raise ValueError(
            "Imputer feature count does not match metadata: "
            f"imputer={imputer_feature_count}, "
            f"metadata={feature_count}"
        )

    return metadata
