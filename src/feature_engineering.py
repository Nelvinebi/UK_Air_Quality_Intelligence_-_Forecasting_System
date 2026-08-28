"""
feature_engineering.py

Builds the model-ready feature set from the merged air-quality + weather
data: calendar features, PM2.5 lags/rolling windows, weather interaction
terms, and the exact next-hour PM2.5 target (matched on Station +
Target_Datetime, not row order).

Direct extraction of notebooks/03_feature_engineering.ipynb into reusable
functions, plus the FEATURES/TARGET constants that train.py and evaluate.py
both import so the three stages never drift out of sync with each other.

Usage (from the project root):
    python -m src.feature_engineering
"""

import logging
from pathlib import Path

import pandas as pd

from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "london_air_quality_weather_2021_2024.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "london_air_quality_features_2021_2024.csv"

TARGET = "PM2.5_target"

FEATURES = [
    "NO2",
    "PM10",
    "PM2.5",
    "O3",
    "Temperature",
    "Dewpoint",
    "Humidity",
    "WindSpeed",
    "WindDirection",
    "Pressure",
    "Visibility",
    "Year",
    "Month",
    "Day",
    "Hour",
    "DayOfWeek",
    "DayOfYear",
    "WeekOfYear",
    "IsWeekend",
    "PM2.5_lag_1h",
    "PM2.5_lag_3h",
    "PM2.5_lag_6h",
    "PM2.5_lag_24h",
    "PM2.5_rolling_3h",
    "PM2.5_rolling_6h",
    "PM2.5_rolling_24h",
    "Temperature_Humidity",
    "WindSpeed_Squared",
    "Pressure_Change",
]

LAG_HOURS = [1, 3, 6, 24]
ROLLING_WINDOWS = {"3h": "PM2.5_rolling_3h", "6h": "PM2.5_rolling_6h", "24h": "PM2.5_rolling_24h"}


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def load_merged_dataset(path: Path = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df


def sort_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["Station", "Datetime"])
        .drop_duplicates(["Station", "Datetime"])
        .reset_index(drop=True)
    )


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = df["Datetime"].dt.year
    df["Month"] = df["Datetime"].dt.month
    df["Day"] = df["Datetime"].dt.day
    df["Hour"] = df["Datetime"].dt.hour
    df["DayOfWeek"] = df["Datetime"].dt.dayofweek
    df["DayOfYear"] = df["Datetime"].dt.dayofyear
    df["WeekOfYear"] = df["Datetime"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Season"] = df["Month"].map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
        }
    )
    return df


def add_lag_features(df: pd.DataFrame, lag_hours=None) -> pd.DataFrame:
    """Exact per-station PM2.5 lags, matched on (Station, Datetime - h),
    not on row position — the dataset has gaps, so shifting rows would
    silently misalign lags across missing hours."""
    if lag_hours is None:
        lag_hours = LAG_HOURS

    df = df.copy()
    pm25_lookup = df[["Station", "Datetime", "PM2.5"]].copy()

    for hours in lag_hours:
        lag_lookup = pm25_lookup.rename(
            columns={"Datetime": "Lag_Datetime", "PM2.5": f"PM2.5_lag_{hours}h"}
        )
        df["Lag_Datetime"] = df["Datetime"] - pd.Timedelta(hours=hours)
        df = df.merge(
            lag_lookup,
            left_on=["Station", "Lag_Datetime"],
            right_on=["Station", "Lag_Datetime"],
            how="left",
            validate="one_to_one",
        ).drop(columns=["Lag_Datetime"])

    return df


def add_rolling_features(df: pd.DataFrame, windows=None) -> pd.DataFrame:
    """Per-station rolling PM2.5 means over trailing time windows.
    `closed="left"` excludes the current row, so these never leak the
    value being predicted."""
    if windows is None:
        windows = ROLLING_WINDOWS

    df = df.copy()
    for window, col_name in windows.items():
        roll_df = (
            df[["Station", "Datetime", "PM2.5"]]
            .sort_values(["Station", "Datetime"])
            .set_index("Datetime")
            .groupby("Station")["PM2.5"]
            .rolling(window, closed="left")
            .mean()
            .rename(col_name)
            .reset_index()
        )
        df = df.merge(roll_df, on=["Station", "Datetime"], how="left", validate="one_to_one")

    return df


def add_weather_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Temperature_Humidity"] = df["Temperature"] * df["Humidity"]
    df["WindSpeed_Squared"] = df["WindSpeed"] ** 2
    df["Pressure_Change"] = df.groupby("Station")["Pressure"].diff()
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Exact next-hour PM2.5 target, matched on (Station, Datetime + 1h) —
    NOT df.shift(-1), which silently breaks whenever a station has a gap
    in its hourly readings."""
    df = df.copy()
    target_lookup = df[["Station", "Datetime", "PM2.5"]].rename(columns={"PM2.5": TARGET})

    df["Target_Datetime"] = df["Datetime"] + pd.Timedelta(hours=1)
    df = df.merge(
        target_lookup.rename(columns={"Datetime": "Target_Datetime"}),
        on=["Station", "Target_Datetime"],
        how="left",
        validate="one_to_one",
    )
    df = df.drop(columns=["Target_Datetime"])
    return df


def validate_target_alignment(df: pd.DataFrame) -> int:
    """Independently recompute the expected next-hour target and count
    mismatches. Should always be 0 — this is the check that caught the
    original shift(-1) bug."""
    validation = df.loc[df[TARGET].notna(), ["Station", "Datetime", TARGET]].copy()
    expected = df[["Station", "Datetime", "PM2.5"]].rename(
        columns={"Datetime": "Target_Datetime", "PM2.5": "Expected_target"}
    )
    validation["Target_Datetime"] = validation["Datetime"] + pd.Timedelta(hours=1)
    validation = validation.merge(
        expected, on=["Station", "Target_Datetime"], how="left", validate="one_to_one"
    )
    return int((validation[TARGET] != validation["Expected_target"]).sum())


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no valid next-hour target and rows with no current
    PM2.5 reading — both are unusable for training."""
    df = df.dropna(subset=[TARGET]).copy()
    df = df.dropna(subset=["PM2.5"]).copy()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline on a merged air-quality +
    weather dataframe and return the model-ready dataset."""
    df = sort_and_dedupe(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_weather_interaction_features(df)
    df = add_target(df)

    alignment_errors = validate_target_alignment(df)
    if alignment_errors != 0:
        raise ValueError(f"Target alignment errors detected: {alignment_errors}")

    df = drop_incomplete_rows(df)

    duplicates = df.duplicated(["Station", "Datetime"]).sum()
    if duplicates != 0:
        raise ValueError(f"Duplicate Station-Datetime rows detected: {duplicates}")

    return df


# ---------------------------------------------------------------------------
# Train/test split + model matrices
# ---------------------------------------------------------------------------


def train_test_split_by_year(df: pd.DataFrame, train_years_max: int = 2023, test_year: int = 2024):
    """Time-based split: everything through `train_years_max` is training
    data, `test_year` is held out entirely. Never randomly shuffled —
    this is a forecasting problem, so the test set must be strictly later
    in time than anything the model trained on."""
    train_df = df[df["Datetime"].dt.year <= train_years_max].copy()
    test_df = df[df["Datetime"].dt.year == test_year].copy()

    if train_df["Datetime"].max() >= test_df["Datetime"].min():
        raise ValueError("Training and testing periods overlap.")

    return train_df, test_df


def get_model_matrices(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features=None, target: str = TARGET
):
    if features is None:
        features = FEATURES

    required_columns = ["Datetime", "Station", target] + features
    missing = [c for c in required_columns if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df[target].copy()
    y_test = test_df[target].copy()

    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Missing target values detected.")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    save: bool = True,
) -> pd.DataFrame:
    logger.info(
        "feature_engineering_started input_path=%s output_path=%s save=%s",
        input_path,
        output_path,
        save,
    )

    df = load_merged_dataset(input_path)

    logger.info(
        "merged_dataset_loaded rows=%d columns=%d",
        df.shape[0],
        df.shape[1],
    )

    df = engineer_features(df)

    logger.info(
        ("features_engineered rows=%d columns=%d stations=%d missing_targets=%d"),
        df.shape[0],
        df.shape[1],
        df["Station"].nunique(),
        df[TARGET].isna().sum(),
    )

    if save:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        df.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            "engineered_dataset_saved path=%s",
            output_path,
        )

    logger.info("feature_engineering_completed")

    return df


if __name__ == "__main__":
    configure_logging()
    run_pipeline()
