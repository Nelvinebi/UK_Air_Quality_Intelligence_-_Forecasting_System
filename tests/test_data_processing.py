from pathlib import Path

import pandas as pd

from src.data_processing import (
    clean_aurn,
    load_all_weather,
    load_aurn_raw,
    merge_air_quality_weather,
    process_weather_file,
    reshape_stations,
)

FIXTURES = Path(__file__).parent / "fixtures"
AURN_FIXTURE = FIXTURES / "aurn_sample.csv"
WEATHER_FIXTURE = FIXTURES / "weather_sample.csv"

EXPECTED_WEATHER_COLUMNS = [
    "Datetime",
    "Temperature",
    "Dewpoint",
    "Humidity",
    "WindSpeed",
    "WindDirection",
    "Pressure",
    "Visibility",
]


def _clean_aurn_fixture():
    raw = load_aurn_raw(AURN_FIXTURE)
    reshaped = reshape_stations(raw)
    return clean_aurn(reshaped)


def _clean_weather_fixture():
    return load_all_weather({2021: WEATHER_FIXTURE})


def test_load_aurn_raw_from_fixture():
    raw = load_aurn_raw(AURN_FIXTURE)

    assert raw.shape == (6, 44)
    assert list(raw.columns[:2]) == ["Date", "Time"]
    assert raw.columns[36] == "London Harlington Nitrogen dioxide"
    assert raw.columns[37] == "London Harlington PM10 particulate matter"
    assert raw.columns[38] == "London Harlington PM2.5 particulate matter"
    assert raw.columns[39] == "London Harlington Ozone"


def test_reshape_stations_extracts_harlington_pollutants():
    raw = load_aurn_raw(AURN_FIXTURE)
    reshaped = reshape_stations(raw)

    harlington = reshaped.loc[
        reshaped["Station"] == "London Harlington"
    ].reset_index(drop=True)

    assert len(harlington) == 6
    assert harlington.loc[0, "NO2"] == 24.0
    assert harlington.loc[0, "PM10"] == 18.0
    assert harlington.loc[0, "PM2.5"] == 12.0
    assert harlington.loc[0, "O3"] == 30.0


def test_clean_aurn_removes_duplicate_station_datetimes():
    cleaned = _clean_aurn_fixture()

    duplicate_count = cleaned.duplicated(
        subset=["Datetime", "Station"]
    ).sum()

    assert duplicate_count == 0
    assert len(cleaned) == 25
    assert cleaned["Station"].nunique() == 5


def test_clean_aurn_converts_negative_pollutants_to_missing():
    cleaned = _clean_aurn_fixture()

    row = cleaned.loc[
        (cleaned["Station"] == "London Harlington")
        & (cleaned["Datetime"] == pd.Timestamp("2021-01-01 02:00:00"))
    ].iloc[0]

    assert pd.isna(row["PM2.5"])
    assert (cleaned["PM2.5"].dropna() >= 0).all()


def test_clean_aurn_preserves_expected_harlington_rows():
    cleaned = _clean_aurn_fixture()

    harlington = cleaned.loc[
        cleaned["Station"] == "London Harlington"
    ].sort_values("Datetime")

    assert len(harlington) == 5
    assert harlington["Datetime"].min() == pd.Timestamp(
        "2021-01-01 00:00:00"
    )
    assert harlington["Datetime"].max() == pd.Timestamp(
        "2021-01-01 04:00:00"
    )


def test_process_weather_file_from_fixture():
    weather = process_weather_file(WEATHER_FIXTURE)

    assert weather.shape == (7, 8)
    assert weather.columns.tolist() == EXPECTED_WEATHER_COLUMNS


def test_process_weather_file_parses_invalid_and_missing_values():
    weather = process_weather_file(WEATHER_FIXTURE)

    assert weather["Datetime"].isna().sum() == 1

    row = weather.loc[
        weather["Datetime"] == pd.Timestamp("2021-01-01 02:00:00")
    ].iloc[0]

    assert pd.isna(row["Humidity"])


def test_load_all_weather_removes_invalid_and_duplicate_timestamps():
    weather = _clean_weather_fixture()

    assert len(weather) == 5
    assert weather["Datetime"].isna().sum() == 0
    assert weather.duplicated(subset=["Datetime"]).sum() == 0


def test_load_all_weather_adds_year():
    weather = _clean_weather_fixture()

    assert "Year" in weather.columns
    assert weather["Year"].eq(2021).all()


def test_merge_air_quality_weather_preserves_station_hour_rows():
    aurn = _clean_aurn_fixture()
    weather = _clean_weather_fixture()

    merged = merge_air_quality_weather(aurn, weather)

    assert len(merged) == 25
    assert merged["Station"].nunique() == 5
    assert merged.duplicated(subset=["Datetime", "Station"]).sum() == 0
    assert merged["Temperature"].notna().sum() == 25


def test_merge_air_quality_weather_propagates_hourly_weather_to_stations():
    aurn = _clean_aurn_fixture()
    weather = _clean_weather_fixture()

    merged = merge_air_quality_weather(aurn, weather)

    midnight = merged.loc[
        merged["Datetime"] == pd.Timestamp("2021-01-01 00:00:00")
    ]

    assert len(midnight) == 5
    assert midnight["Temperature"].eq(5.0).all()
    assert midnight["Humidity"].eq(85.0).all()


def test_merge_preserves_missing_hourly_weather_values():
    aurn = _clean_aurn_fixture()
    weather = _clean_weather_fixture()

    merged = merge_air_quality_weather(aurn, weather)

    two_am = merged.loc[
        merged["Datetime"] == pd.Timestamp("2021-01-01 02:00:00")
    ]

    assert len(two_am) == 5
    assert two_am["Humidity"].isna().sum() == 5
