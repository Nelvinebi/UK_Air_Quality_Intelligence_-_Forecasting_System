"""
evaluate.py

Loads the saved final model and produces the analysis that was still
outstanding per the project status notes: feature importance, residual /
error analysis, station-level evaluation, and the final set of figures.

Usage (from the project root):
    python -m src.evaluate
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.feature_engineering import (
    FEATURES,
    TARGET,
    get_model_matrices,
    train_test_split_by_year,
)
from src.train import DATA_PATH, IMPUTER_PATH, MODEL_PATH, load_engineered_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

PREDICTIONS_PATH = REPORTS_DIR / "final_predictions.csv"
STATION_ERRORS_PATH = REPORTS_DIR / "station_level_errors.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"


def load_model_and_imputer(model_path: Path = MODEL_PATH, imputer_path: Path = IMPUTER_PATH):
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    return model, imputer


def compute_feature_importance(model, features=None) -> pd.DataFrame:
    if features is None:
        features = FEATURES
    return (
        pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def build_prediction_df(test_df: pd.DataFrame, y_test: pd.Series, predictions) -> pd.DataFrame:
    prediction_df = pd.DataFrame(
        {
            "Datetime": test_df["Datetime"].values,
            "Station": test_df["Station"].values,
            "Actual": y_test.values,
            "Predicted": predictions,
        }
    )
    prediction_df["Residual"] = prediction_df["Actual"] - prediction_df["Predicted"]
    prediction_df["Absolute_Error"] = prediction_df["Residual"].abs()

    if len(prediction_df) != len(test_df):
        raise ValueError("Prediction/test alignment failed.")

    return prediction_df


def error_summary(prediction_df: pd.DataFrame) -> dict:
    return {
        "Mean_Absolute_Error": prediction_df["Absolute_Error"].mean(),
        "Mean_Residual": prediction_df["Residual"].mean(),
        "Median_Absolute_Error": prediction_df["Absolute_Error"].median(),
        "Max_Absolute_Error": prediction_df["Absolute_Error"].max(),
    }


def station_level_errors(prediction_df: pd.DataFrame) -> pd.DataFrame:
    return (
        prediction_df.groupby("Station")
        .agg(
            MAE=("Absolute_Error", "mean"),
            RMSE=("Residual", lambda x: np.sqrt(np.mean(x ** 2))),
            Mean_Residual=("Residual", "mean"),
            Observations=("Residual", "size"),
        )
        .sort_values("RMSE", ascending=False)
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_feature_importance(feature_importance: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    top_features = feature_importance.head(top_n).sort_values("Importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(prediction_df: pd.DataFrame, path: Path, n_points: int = 500) -> None:
    plot_data = prediction_df.iloc[:n_points]
    plt.figure(figsize=(12, 5))
    plt.plot(plot_data["Actual"].to_numpy(), label="Actual")
    plt.plot(plot_data["Predicted"].to_numpy(), label="Random Forest")
    plt.xlabel("Test observation")
    plt.ylabel("PM2.5")
    plt.title("Actual vs Predicted PM2.5 (Final Model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_residual_distribution(prediction_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(prediction_df["Residual"], bins=50)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution (Final Model)")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_station_errors(station_errors: pd.DataFrame, path: Path) -> None:
    ordered = station_errors.sort_values("RMSE")
    plt.figure(figsize=(9, 5))
    plt.barh(ordered.index, ordered["RMSE"])
    plt.xlabel("RMSE")
    plt.ylabel("Station")
    plt.title("RMSE by Station (Final Model)")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(data_path: Path = DATA_PATH, save: bool = True) -> dict:
    print("Loading model, imputer, and data...")
    model, imputer = load_model_and_imputer()
    df = load_engineered_dataset(data_path)

    train_df, test_df = train_test_split_by_year(df)
    X_train, X_test, y_train, y_test = get_model_matrices(train_df, test_df)
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test), columns=FEATURES, index=X_test.index
    )

    print("Generating predictions...")
    predictions = model.predict(X_test_imputed)
    prediction_df = build_prediction_df(test_df, y_test, predictions)

    print("Computing feature importance...")
    feature_importance = compute_feature_importance(model)
    print(feature_importance.head(15).to_string(index=False))

    print("\nError analysis:")
    errors = error_summary(prediction_df)
    for key, value in errors.items():
        print(f"  {key}: {round(value, 3)}")

    print("\nLargest prediction errors:")
    largest_errors = prediction_df.nlargest(10, "Absolute_Error")
    print(
        largest_errors[
            ["Datetime", "Station", "Actual", "Predicted", "Residual", "Absolute_Error"]
        ].to_string(index=False)
    )

    print("\nStation-level evaluation:")
    station_errors = station_level_errors(prediction_df)
    print(station_errors.round(3).to_string())

    if save:
        print("\nSaving figures and reports...")
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        plot_feature_importance(feature_importance, FIGURES_DIR / "final_feature_importance.png")
        plot_actual_vs_predicted(prediction_df, FIGURES_DIR / "final_actual_vs_predicted.png")
        plot_residual_distribution(prediction_df, FIGURES_DIR / "final_residual_distribution.png")
        plot_station_errors(station_errors, FIGURES_DIR / "final_station_rmse.png")

        prediction_df.to_csv(PREDICTIONS_PATH, index=False)
        station_errors.to_csv(STATION_ERRORS_PATH)
        feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

        print(f"  Saved figures to: {FIGURES_DIR}")
        print(f"  Saved: {PREDICTIONS_PATH}")
        print(f"  Saved: {STATION_ERRORS_PATH}")
        print(f"  Saved: {FEATURE_IMPORTANCE_PATH}")

    return {
        "errors": errors,
        "feature_importance": feature_importance,
        "station_errors": station_errors,
    }


if __name__ == "__main__":
    run_evaluation()
