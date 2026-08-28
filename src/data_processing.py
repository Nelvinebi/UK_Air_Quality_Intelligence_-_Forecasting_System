"""
data_processing.py

Loads and cleans the raw AURN air-quality data and MIDAS Heathrow weather
data, merges them, and writes the processed datasets to data/processed/.

This is a direct extraction of the working logic from
notebooks/01_data_collection_and_cleaning.ipynb, refactored into reusable
functions so it can be called from other scripts (train.py, the Streamlit
app, etc.) instead of being copy-pasted.

Usage (from the project root):
    python -m src.data_processing
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

AURN_FILE = RAW_DIR / "94529965511.csv"
WEATHER_FILES = {
    2021: RAW_DIR / "heathrow_2021.csv",
    2022: RAW_DIR / "heathrow_2022.csv",
    2023: RAW_DIR / "heathrow_2023.csv",
    2024: RAW_DIR / "heathrow_2024.csv",
}

AURN_OUTPUT = PROCESSED_DIR / "london_aurn_pm25.csv"
WEATHER_OUTPUT = PROCESSED_DIR / "heathrow_weather_2021_2024.csv"
MERGED_OUTPUT = PROCESSED_DIR / "london_air_quality_weather_2021_2024.csv"

START_DATE = "2021-01-01 00:00:00"
END_DATE = "2024-12-31 23:00:00"

# ---------------------------------------------------------------------------
# AURN station layout
# ---------------------------------------------------------------------------

# Column offset -> station name, as laid out in the UK-AIR export.
STATIONS = {
    2: "London Bexley",
    8: "London Bloomsbury",
    16: "London Eltham",
    24: "London Farringdon Street",
    28: "London Haringey Priory Park South",
    36: "London Harlington",
    44: "London Hillingdon",
    52: "London Honor Oak Park",
    58: "London Marylebone Road",
    66: "London N. Kensington",
    74: "London Norbury Manor School",
    78: "London Teddington Bushy Park",
    82: "London Westminster",
}

# The five stations with the most complete, reliable data — used for modeling.
SELECTED_STATIONS = [
    "London Harlington",
    "London N. Kensington",
    "London Honor Oak Park",
    "London Marylebone Road",
    "London Bloomsbury",
]

POLLUTANTS = {
    "NO2": "Nitrogen dioxide",
    "PM10": "PM10 particulate matter",
    "PM2.5": "PM2.5 particulate matter",
    "O3": "Ozone",
}
POLLUTANT_COLUMNS = list(POLLUTANTS.keys())


def get_pollutant_column(df: pd.DataFrame, station_start: int, pollutant: str):
    """Find the first column containing `pollutant` within a station's
    8-column block, starting at `station_start`."""
    station_end = min(station_start + 8, len(df.columns))
    for col_idx in range(station_start, station_end):
        if pollutant in str(df.columns[col_idx]):
            return col_idx
    return None


# ---------------------------------------------------------------------------
# AURN loading / cleaning
# ---------------------------------------------------------------------------


def load_aurn_raw(aurn_file: Path = AURN_FILE) -> pd.DataFrame:
    """Load the raw UK-AIR AURN export. The first 17 rows are metadata,
    so row 18 becomes the header."""
    return pd.read_csv(aurn_file, skiprows=17, low_memory=False)


def reshape_stations(aurn_raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide, multi-station AURN export into a long
    (Datetime, Station, pollutant...) table."""
    records = []
    for station_start, station_name in STATIONS.items():
        station_data = pd.DataFrame(
            {
                "Date": aurn_raw["Date"],
                "Time": aurn_raw["Time"],
                "Station": station_name,
            }
        )
        for pollutant, search_name in POLLUTANTS.items():
            col_idx = get_pollutant_column(aurn_raw, station_start, search_name)
            if col_idx is not None:
                station_data[pollutant] = pd.to_numeric(aurn_raw.iloc[:, col_idx], errors="coerce")
            else:
                station_data[pollutant] = np.nan
        records.append(station_data)

    aurn_clean = pd.concat(records, ignore_index=True)

    aurn_clean["Datetime"] = pd.to_datetime(
        aurn_clean["Date"].astype(str) + " " + aurn_clean["Time"].astype(str),
        errors="coerce",
    )
    aurn_clean = aurn_clean.drop(columns=["Date", "Time"])
    aurn_clean = aurn_clean[["Datetime", "Station"] + POLLUTANT_COLUMNS]

    return aurn_clean


def clean_aurn(
    aurn_clean: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    selected_stations=None,
) -> pd.DataFrame:
    """Drop invalid timestamps/duplicates, remove negative pollutant
    readings (treated as missing, not clipped to zero), restrict to the
    study period, and keep only the selected high-quality stations."""
    if selected_stations is None:
        selected_stations = SELECTED_STATIONS

    aurn_clean = aurn_clean.dropna(subset=["Datetime"])
    aurn_clean = aurn_clean.drop_duplicates(subset=["Datetime", "Station"])

    aurn_clean = aurn_clean[
        (aurn_clean["Datetime"] >= start_date) & (aurn_clean["Datetime"] <= end_date)
    ].copy()

    # Negative concentrations are sensor artifacts, not real readings.
    for pollutant in POLLUTANT_COLUMNS:
        aurn_clean.loc[aurn_clean[pollutant] < 0, pollutant] = np.nan

    aurn_model = aurn_clean[aurn_clean["Station"].isin(selected_stations)].copy()

    return aurn_model


# ---------------------------------------------------------------------------
# Weather loading / cleaning
# ---------------------------------------------------------------------------


def process_weather_file(file_path: Path) -> pd.DataFrame:
    """Load and standardize a single MIDAS Heathrow hourly weather file."""
    weather = pd.read_csv(file_path, skiprows=283, na_values="NA")

    weather = weather[
        [
            "ob_time",
            "air_temperature",
            "dewpoint",
            "rltv_hum",
            "wind_speed",
            "wind_direction",
            "msl_pressure",
            "visibility",
        ]
    ].copy()

    weather = weather.rename(
        columns={
            "ob_time": "Datetime",
            "air_temperature": "Temperature",
            "dewpoint": "Dewpoint",
            "rltv_hum": "Humidity",
            "wind_speed": "WindSpeed",
            "wind_direction": "WindDirection",
            "msl_pressure": "Pressure",
            "visibility": "Visibility",
        }
    )

    weather["Datetime"] = pd.to_datetime(weather["Datetime"], errors="coerce")
    return weather


def load_all_weather(
    weather_files=None, start_date: str = START_DATE, end_date: str = END_DATE
) -> pd.DataFrame:
    """Load and concatenate all yearly weather files, then clean them."""
    if weather_files is None:
        weather_files = WEATHER_FILES

    weather_records = []
    for year, file_path in weather_files.items():
        yearly_weather = process_weather_file(file_path)
        yearly_weather["Year"] = year
        weather_records.append(yearly_weather)

    weather_all = pd.concat(weather_records, ignore_index=True)

    weather_all = weather_all.dropna(subset=["Datetime"])
    weather_all = weather_all.drop_duplicates(subset=["Datetime"])
    weather_all = weather_all[
        (weather_all["Datetime"] >= start_date) & (weather_all["Datetime"] <= end_date)
    ].copy()

    return weather_all


# ---------------------------------------------------------------------------
# Merge + save
# ---------------------------------------------------------------------------


def merge_air_quality_weather(aurn_model: pd.DataFrame, weather_all: pd.DataFrame) -> pd.DataFrame:
    """Left-merge pollution readings onto weather by exact hourly timestamp."""
    weather_for_merge = weather_all.drop(columns=["Year"], errors="ignore")
    merged_df = aurn_model.merge(weather_for_merge, on="Datetime", how="left")
    return merged_df


def save_processed_datasets(
    aurn_model: pd.DataFrame,
    weather_all: pd.DataFrame,
    merged_df: pd.DataFrame,
    processed_dir: Path = PROCESSED_DIR,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    aurn_model.to_csv(AURN_OUTPUT, index=False)
    weather_all.to_csv(WEATHER_OUTPUT, index=False)
    merged_df.to_csv(MERGED_OUTPUT, index=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    save: bool = True,
) -> pd.DataFrame:
    """Run the full collection/cleaning pipeline end to end."""
    logger.info(
        "data_processing_started raw_dir=%s processed_dir=%s save=%s",
        raw_dir,
        processed_dir,
        save,
    )

    aurn_raw = load_aurn_raw(
        raw_dir / AURN_FILE.name,
    )

    logger.info(
        "aurn_raw_loaded rows=%d columns=%d",
        aurn_raw.shape[0],
        aurn_raw.shape[1],
    )

    aurn_long = reshape_stations(aurn_raw)

    logger.info(
        "aurn_stations_reshaped rows=%d",
        len(aurn_long),
    )

    aurn_model = clean_aurn(aurn_long)

    logger.info(
        "aurn_cleaned rows=%d stations=%d",
        len(aurn_model),
        aurn_model["Station"].nunique(),
    )

    weather_files = {year: raw_dir / path.name for year, path in WEATHER_FILES.items()}
    weather_all = load_all_weather(weather_files)

    logger.info(
        "weather_loaded rows=%d",
        len(weather_all),
    )

    merged_df = merge_air_quality_weather(
        aurn_model,
        weather_all,
    )

    logger.info(
        "air_quality_weather_merged rows=%d columns=%d",
        merged_df.shape[0],
        merged_df.shape[1],
    )

    if save:
        save_processed_datasets(
            aurn_model,
            weather_all,
            merged_df,
            processed_dir,
        )

        logger.info(
            ("processed_datasets_saved aurn_path=%s weather_path=%s merged_path=%s"),
            AURN_OUTPUT,
            WEATHER_OUTPUT,
            MERGED_OUTPUT,
        )

    logger.info("data_processing_completed")

    return merged_df


if __name__ == "__main__":
    configure_logging()
    run_pipeline()
