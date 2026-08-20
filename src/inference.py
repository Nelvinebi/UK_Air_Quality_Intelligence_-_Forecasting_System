from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def validate_features(
    data: pd.DataFrame,
    features: Iterable[str],
) -> list[str]:
    """
    Validate that all required model features exist in the input DataFrame.

    Returns the ordered feature list used by the model.
    """
    feature_list = list(features)

    missing = [
        feature
        for feature in feature_list
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required model features: {missing}"
        )

    return feature_list


def prepare_features(
    data: pd.DataFrame,
    imputer,
    features: Iterable[str],
) -> pd.DataFrame:
    """
    Select model features, apply the fitted imputer, and restore
    the feature names expected by the trained estimator.
    """
    feature_list = validate_features(data, features)

    X = data.loc[:, feature_list].copy()

    transformed = imputer.transform(X)

    return pd.DataFrame(
        transformed,
        columns=feature_list,
        index=X.index,
    )


def predict(
    model,
    imputer,
    data: pd.DataFrame,
    features: Iterable[str],
) -> np.ndarray:
    """
    Generate model predictions from validated input data.
    """
    X = prepare_features(
        data=data,
        imputer=imputer,
        features=features,
    )

    predictions = model.predict(X)

    return np.asarray(predictions)


def predict_row(
    model,
    imputer,
    row: pd.Series,
    features: Iterable[str],
) -> float:
    """
    Generate a single prediction from one pandas Series.
    """
    frame = row.to_frame().T

    predictions = predict(
        model=model,
        imputer=imputer,
        data=frame,
        features=features,
    )

    return float(predictions[0])
