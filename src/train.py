"""
train.py

Trains the project's final Random Forest model and saves it (plus the
fitted imputer and metadata) to models/.

Model choice: hyperparameter tuning (see notebooks/05_model_tuning.ipynb)
did not beat the original baseline configuration, so the "final" model
IS that baseline config — RandomForestRegressor(n_estimators=100,
random_state=42, n_jobs=-1) — trained on the full training set. This
matches models/random_forest_pm25.pkl as saved by the notebook.

Usage (from the project root):
    python -m src.train
"""

import json
import logging
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.feature_engineering import (
    FEATURES,
    TARGET,
    get_model_matrices,
    train_test_split_by_year,
)
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "london_air_quality_features_2021_2024.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest_pm25.pkl"
IMPUTER_PATH = MODEL_DIR / "imputer.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "baseline_model_results.csv"

# The final model's configuration. Tuning (RandomizedSearchCV, then a
# lightweight grid) never beat the unbounded-depth baseline on accuracy, but
# capping max_depth=15 matches it (MAE 1.241 vs 1.249, same RMSE/R2) while
# cutting the saved model from ~206MB to ~23MB — well under GitHub's 100MB
# limit, so the model can just live in the repo instead of needing git-lfs.
FINAL_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "random_state": 42,
    "n_jobs": -1,
}


def load_engineered_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df


def impute_features(X_train: pd.DataFrame, X_test: pd.DataFrame, features=None):
    if features is None:
        features = FEATURES

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=features, index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test), columns=features, index=X_test.index
    )
    return imputer, X_train_imputed, X_test_imputed


def train_random_forest(X_train_imputed: pd.DataFrame, y_train: pd.Series, **params) -> RandomForestRegressor:
    model_params = {**FINAL_MODEL_PARAMS, **params}
    model = RandomForestRegressor(**model_params)
    model.fit(X_train_imputed, y_train)
    return model


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def save_model(model, imputer, features, target, path: Path = MODEL_PATH) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # compress=3 keeps a 100-tree, unbounded-depth forest to ~200MB instead
    # of ~900MB uncompressed. Still large — see README for the git-lfs note.
    joblib.dump(model, path, compress=3)
    joblib.dump(imputer, IMPUTER_PATH, compress=3)

    metadata = {
        "features": features,
        "target": target,
        "model_params": FINAL_MODEL_PARAMS,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_training(data_path: Path = DATA_PATH, save: bool = True) -> dict:
    logger.info(
        "training_started data_path=%s save=%s",
        data_path,
        save,
    )

    df = load_engineered_dataset(data_path)

    logger.info(
        "training_dataset_loaded rows=%d columns=%d",
        df.shape[0],
        df.shape[1],
    )

    train_df, test_df = train_test_split_by_year(df)
    X_train, X_test, y_train, y_test = get_model_matrices(
        train_df,
        test_df,
    )

    logger.info(
        (
            "training_split_completed train_rows=%d test_rows=%d "
            "feature_count=%d"
        ),
        len(train_df),
        len(test_df),
        X_train.shape[1],
    )

    imputer, X_train_imputed, X_test_imputed = impute_features(
        X_train,
        X_test,
    )

    logger.info(
        "feature_imputation_completed strategy=median"
    )

    logger.info(
        "random_forest_training_started params=%s",
        FINAL_MODEL_PARAMS,
    )

    model = train_random_forest(
        X_train_imputed,
        y_train,
    )

    logger.info("random_forest_training_completed")

    predictions = model.predict(X_test_imputed)
    metrics = evaluate_predictions(
        y_test,
        predictions,
    )

    logger.info(
        "training_metrics MAE=%.3f RMSE=%.3f R2=%.3f",
        metrics["MAE"],
        metrics["RMSE"],
        metrics["R2"],
    )

    if save:
        save_model(
            model,
            imputer,
            FEATURES,
            TARGET,
        )

        logger.info(
            (
                "model_artifacts_saved model_path=%s "
                "imputer_path=%s metadata_path=%s"
            ),
            MODEL_PATH,
            IMPUTER_PATH,
            METADATA_PATH,
        )

        results_row = pd.DataFrame(
            [
                {
                    "Model": "Random Forest (final)",
                    **metrics,
                }
            ]
        )

        if RESULTS_PATH.exists():
            existing = pd.read_csv(RESULTS_PATH)
            existing = existing[
                existing["Model"] != "Random Forest (final)"
            ]
            results = pd.concat(
                [existing, results_row],
                ignore_index=True,
            )
        else:
            results = results_row

        RESULTS_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        results.to_csv(
            RESULTS_PATH,
            index=False,
        )

        logger.info(
            "training_results_saved path=%s",
            RESULTS_PATH,
        )

    logger.info("training_completed")

    return metrics


if __name__ == "__main__":
    configure_logging()
    run_training()
