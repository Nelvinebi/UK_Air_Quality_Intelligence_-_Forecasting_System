"""Raw external-data schema validation for AURN and MIDAS inputs."""

from __future__ import annotations

import pandas as pd

from src.validation import validate_required_columns

AURN_REQUIRED_COLUMNS = ("Date", "Time")
AURN_POLLUTANT_PATTERNS = (
    "Nitrogen dioxide",
    "PM10 particulate matter",
    "PM2.5 particulate matter",
    "Ozone",
)

MIDAS_REQUIRED_COLUMNS = (
    "ob_time",
    "air_temperature",
    "dewpoint",
    "rltv_hum",
    "wind_speed",
    "wind_direction",
    "msl_pressure",
    "visibility",
)
MIDAS_NUMERIC_COLUMNS = MIDAS_REQUIRED_COLUMNS[1:]

MODEL_START = pd.Timestamp("2021-01-01 00:00:00")
MODEL_END = pd.Timestamp("2024-12-31 23:00:00")


class RawDataValidationError(ValueError):
    """Raised when an external raw-data file does not match the expected schema."""


def _require_columns(
    df: pd.DataFrame,
    required_columns: tuple[str, ...],
    *,
    context: str,
) -> None:
    """Translate shared required-column validation into a raw-data error."""
    try:
        validate_required_columns(df, required_columns, context=context)
    except ValueError as exc:
        raise RawDataValidationError(str(exc)) from exc


def validate_aurn_raw(df: pd.DataFrame) -> None:
    """Validate the structural contract of a raw UK-AIR AURN export."""
    if df.empty:
        raise RawDataValidationError("AURN raw dataset is empty.")

    _require_columns(df, AURN_REQUIRED_COLUMNS, context="AURN raw data")

    parsed_datetime = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce",
        format="mixed",
    )
    if not parsed_datetime.notna().any():
        raise RawDataValidationError("AURN raw data contains no parseable Date/Time values.")

    usable = parsed_datetime.dropna()
    if usable.max() < MODEL_START or usable.min() > MODEL_END:
        raise RawDataValidationError(
            "AURN raw data does not overlap the project modelling period 2021-2024."
        )

    headers = [str(column) for column in df.columns]
    missing_pollutants = [
        pollutant
        for pollutant in AURN_POLLUTANT_PATTERNS
        if not any(pollutant in header for header in headers)
    ]
    if missing_pollutants:
        raise RawDataValidationError(
            f"AURN raw data is missing expected pollutant column patterns: {missing_pollutants}"
        )


def validate_midas_raw(df: pd.DataFrame) -> None:
    """Validate required MIDAS columns, timestamps, and numeric field types."""
    if df.empty:
        raise RawDataValidationError("MIDAS raw dataset is empty.")

    _require_columns(df, MIDAS_REQUIRED_COLUMNS, context="MIDAS raw data")

    parsed_datetime = pd.to_datetime(df["ob_time"], errors="coerce", format="mixed")
    if not parsed_datetime.notna().any():
        raise RawDataValidationError("MIDAS raw data contains no parseable ob_time values.")

    usable = parsed_datetime.dropna()
    if usable.max() < MODEL_START or usable.min() > MODEL_END:
        raise RawDataValidationError(
            "MIDAS raw data does not overlap the project modelling period 2021-2024."
        )

    invalid_numeric: dict[str, int] = {}
    for column in MIDAS_NUMERIC_COLUMNS:
        non_null = df[column].notna()
        coerced = pd.to_numeric(df[column], errors="coerce")
        invalid_count = int((non_null & coerced.isna()).sum())
        if invalid_count:
            invalid_numeric[column] = invalid_count

    if invalid_numeric:
        raise RawDataValidationError(
            "MIDAS raw data contains non-numeric values in required numeric columns: "
            f"{invalid_numeric}"
        )
