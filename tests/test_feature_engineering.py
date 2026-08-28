import pandas as pd
import pytest

from src.feature_engineering import (
    TARGET,
    add_lag_features,
    add_rolling_features,
    add_target,
    add_time_features,
    get_model_matrices,
    sort_and_dedupe,
    train_test_split_by_year,
    validate_target_alignment,
)


def make_sample_data():
    """Create a tiny deterministic dataset for feature-engineering tests."""
    return pd.DataFrame(
        {
            "Station": [
                "Station A",
                "Station A",
                "Station A",
                "Station A",
                "Station B",
                "Station B",
                "Station B",
            ],
            "Datetime": pd.to_datetime(
                [
                    "2023-01-01 00:00:00",
                    "2023-01-01 01:00:00",
                    "2023-01-01 02:00:00",
                    "2023-01-01 04:00:00",  # intentional 03:00 gap
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                    "2024-01-01 02:00:00",
                ]
            ),
            "PM2.5": [10.0, 20.0, 30.0, 50.0, 100.0, 110.0, 120.0],
            "NO2": [1.0] * 7,
            "PM10": [2.0] * 7,
            "O3": [3.0] * 7,
            "Temperature": [10.0] * 7,
            "Dewpoint": [5.0] * 7,
            "Humidity": [80.0] * 7,
            "WindSpeed": [2.0] * 7,
            "WindDirection": [180.0] * 7,
            "Pressure": [1000.0] * 7,
            "Visibility": [10.0] * 7,
        }
    )


def test_sort_and_dedupe_removes_duplicate_station_datetime():
    df = make_sample_data()

    duplicate = df.iloc[[0]].copy()
    df = pd.concat([df, duplicate], ignore_index=True)

    result = sort_and_dedupe(df)

    assert result.duplicated(["Station", "Datetime"]).sum() == 0


def test_add_time_features_creates_expected_calendar_features():
    df = make_sample_data()

    result = add_time_features(df)

    expected_columns = {
        "Year",
        "Month",
        "Day",
        "Hour",
        "DayOfWeek",
        "DayOfYear",
        "WeekOfYear",
        "IsWeekend",
        "Season",
    }

    assert expected_columns.issubset(result.columns)

    first_row = result.iloc[0]

    assert first_row["Year"] == 2023
    assert first_row["Month"] == 1
    assert first_row["Day"] == 1
    assert first_row["Hour"] == 0
    assert first_row["IsWeekend"] == 1
    assert first_row["Season"] == "Winter"


def test_one_hour_lag_matches_exact_timestamp():
    df = make_sample_data()

    result = add_lag_features(df, lag_hours=[1])

    row = result[
        (result["Station"] == "Station A")
        & (result["Datetime"] == pd.Timestamp("2023-01-01 01:00:00"))
    ].iloc[0]

    assert row["PM2.5_lag_1h"] == 10.0


def test_lag_does_not_bridge_missing_hour():
    df = make_sample_data()

    result = add_lag_features(df, lag_hours=[1])

    row = result[
        (result["Station"] == "Station A")
        & (result["Datetime"] == pd.Timestamp("2023-01-01 04:00:00"))
    ].iloc[0]

    assert pd.isna(row["PM2.5_lag_1h"])


def test_lag_features_never_cross_station_boundaries():
    df = make_sample_data()

    result = add_lag_features(df, lag_hours=[1])

    row = result[
        (result["Station"] == "Station B")
        & (result["Datetime"] == pd.Timestamp("2024-01-01 01:00:00"))
    ].iloc[0]

    assert row["PM2.5_lag_1h"] == 100.0
    assert row["PM2.5_lag_1h"] != 50.0


def test_rolling_feature_excludes_current_observation():
    df = make_sample_data()

    result = add_rolling_features(
        df,
        windows={"3h": "PM2.5_rolling_3h"},
    )

    row = result[
        (result["Station"] == "Station A")
        & (result["Datetime"] == pd.Timestamp("2023-01-01 02:00:00"))
    ].iloc[0]

    assert row["PM2.5_rolling_3h"] == pytest.approx(15.0)


def test_target_is_exactly_next_hour():
    df = make_sample_data()

    result = add_target(df)

    row = result[
        (result["Station"] == "Station A")
        & (result["Datetime"] == pd.Timestamp("2023-01-01 00:00:00"))
    ].iloc[0]

    assert row[TARGET] == 20.0


def test_target_does_not_bridge_missing_hour():
    df = make_sample_data()

    result = add_target(df)

    row = result[
        (result["Station"] == "Station A")
        & (result["Datetime"] == pd.Timestamp("2023-01-01 02:00:00"))
    ].iloc[0]

    assert pd.isna(row[TARGET])


def test_target_never_crosses_station_boundary():
    df = make_sample_data()

    result = add_target(df)

    row = result[
        (result["Station"] == "Station B")
        & (result["Datetime"] == pd.Timestamp("2024-01-01 00:00:00"))
    ].iloc[0]

    assert row[TARGET] == 110.0


def test_target_alignment_validation_returns_zero_errors():
    df = make_sample_data()
    df = add_target(df)

    errors = validate_target_alignment(df)

    assert errors == 0


def test_train_test_split_is_time_based():
    df = make_sample_data()

    train_df, test_df = train_test_split_by_year(
        df,
        train_years_max=2023,
        test_year=2024,
    )

    assert not train_df.empty
    assert not test_df.empty

    assert train_df["Datetime"].dt.year.max() == 2023
    assert test_df["Datetime"].dt.year.min() == 2024

    assert train_df["Datetime"].max() < test_df["Datetime"].min()


def test_get_model_matrices_raises_for_missing_required_feature():
    df = make_sample_data()

    df["PM2.5_target"] = df["PM2.5"]

    train_df = df[df["Datetime"].dt.year == 2023].copy()
    test_df = df[df["Datetime"].dt.year == 2024].copy()

    with pytest.raises(ValueError, match="Missing required columns"):
        get_model_matrices(
            train_df,
            test_df,
            features=["ThisFeatureDoesNotExist"],
        )
