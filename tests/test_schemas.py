from pathlib import Path

import pandas as pd
import pytest

from src.schemas import RawDataValidationError, validate_aurn_raw, validate_midas_raw

FIXTURES = Path(__file__).parent / "fixtures"
AURN_FIXTURE = FIXTURES / "aurn_sample.csv"
WEATHER_FIXTURE = FIXTURES / "weather_sample.csv"


def _load_aurn_fixture() -> pd.DataFrame:
    return pd.read_csv(AURN_FIXTURE, skiprows=17, low_memory=False)


def _load_weather_fixture() -> pd.DataFrame:
    return pd.read_csv(WEATHER_FIXTURE, skiprows=283, na_values="NA", low_memory=False)


def test_validate_aurn_raw_accepts_cleanable_fixture():
    validate_aurn_raw(_load_aurn_fixture())


def test_validate_aurn_raw_rejects_missing_required_column():
    df = _load_aurn_fixture().drop(columns=["Date"])

    with pytest.raises(RawDataValidationError, match="Missing required columns"):
        validate_aurn_raw(df)


def test_validate_aurn_raw_rejects_missing_pollutant_pattern():
    df = _load_aurn_fixture().drop(columns=["London Harlington Ozone"])

    with pytest.raises(RawDataValidationError, match="Ozone"):
        validate_aurn_raw(df)


def test_validate_aurn_raw_rejects_completely_unusable_timestamps():
    df = _load_aurn_fixture()
    df["Date"] = "not-a-date"
    df["Time"] = "not-a-time"

    with pytest.raises(RawDataValidationError, match="no parseable Date/Time"):
        validate_aurn_raw(df)


def test_validate_midas_raw_accepts_missing_values_and_isolated_bad_timestamp():
    validate_midas_raw(_load_weather_fixture())


def test_validate_midas_raw_rejects_missing_required_column():
    df = _load_weather_fixture().drop(columns=["visibility"])

    with pytest.raises(RawDataValidationError, match="Missing required columns"):
        validate_midas_raw(df)


def test_validate_midas_raw_rejects_non_numeric_required_value():
    df = _load_weather_fixture()
    df["air_temperature"] = df["air_temperature"].astype("object")
    df.loc[df.index[0], "air_temperature"] = "not-numeric"

    with pytest.raises(RawDataValidationError, match="air_temperature"):
        validate_midas_raw(df)


def test_validate_midas_raw_rejects_completely_unusable_timestamps():
    df = _load_weather_fixture()
    df["ob_time"] = "not-a-date"

    with pytest.raises(RawDataValidationError, match="no parseable ob_time"):
        validate_midas_raw(df)


def test_validate_aurn_raw_rejects_data_outside_modelling_window():
    df = _load_aurn_fixture()
    df["Date"] = "2030-01-01"

    with pytest.raises(
        RawDataValidationError,
        match="does not overlap the project modelling period",
    ):
        validate_aurn_raw(df)


def test_validate_midas_raw_rejects_data_outside_modelling_window():
    df = _load_weather_fixture()
    df["ob_time"] = "2030-01-01 00:00:00"

    with pytest.raises(
        RawDataValidationError,
        match="does not overlap the project modelling period",
    ):
        validate_midas_raw(df)


def test_validate_aurn_raw_rejects_empty_dataset():
    with pytest.raises(RawDataValidationError, match="AURN raw dataset is empty"):
        validate_aurn_raw(pd.DataFrame())


def test_validate_midas_raw_rejects_empty_dataset():
    with pytest.raises(RawDataValidationError, match="MIDAS raw dataset is empty"):
        validate_midas_raw(pd.DataFrame())
