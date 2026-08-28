"""
streamlit_app.py

London Air Quality Intelligence — an interactive dashboard for the
one-hour-ahead PM2.5 forecasting model.

Two ways to explore the model:
  - Live Station Snapshot: browse real, held-out 2024 station-hours and
    see the model's forecast against what actually happened.
  - What-if: nudge the current-hour pollutant/weather readings for a
    selected snapshot and see how the forecast responds, while the
    snapshot's real recent history (lags/rolling averages) stays fixed.
    (Fabricating a full 29-feature vector by hand would produce
    meaningless predictions, since those features are highly
    correlated in reality — this keeps every "what-if" physically
    sensible.)

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Project root / import path
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.air_quality import daqi_band
from app.data_loaders import (
    DATA_PATH,
    MODEL_DIR,
    load_model_bundle,
    load_reports,
    load_test_data,
)
from app.diagnostics import render_model_diagnostics
from app.styling import BACKGROUND_IMAGE, inject_style
from app.what_if import render_what_if
from src.inference import predict_row

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


CURRENT_HOUR_INPUTS = [
    "PM2.5",
    "NO2",
    "PM10",
    "O3",
    "Temperature",
    "Humidity",
    "WindSpeed",
]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="London Air Quality Intelligence",
        page_icon="🕰️",
        layout="wide",
    )

    inject_style()

    required_paths = [
        BACKGROUND_IMAGE,
        MODEL_DIR / "random_forest_pm25.pkl",
        MODEL_DIR / "imputer.pkl",
        MODEL_DIR / "model_metadata.json",
        DATA_PATH,
    ]

    missing = [path for path in required_paths if not path.exists()]

    if missing:
        st.error(
            "Missing required files:\n\n"
            + "\n".join(f"- `{path.relative_to(PROJECT_ROOT)}`" for path in missing)
            + "\n\nRun the pipeline first: "
            "`python -m src.data_processing` → "
            "`python -m src.feature_engineering` → "
            "`python -m src.train`."
        )
        return

    model, imputer, metadata = load_model_bundle()

    features = metadata["features"]

    df = load_test_data()
    reports = load_reports()

    st.markdown(
        ('<div class="eyebrow">LONDON · FIVE MONITORING STATIONS · HELD-OUT 2024</div>'),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-title">Air Quality Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="hero-sub">'
            "One-hour-ahead PM2.5 forecasts from a Random Forest trained "
            "on AURN pollution monitoring and Heathrow weather data, "
            "2021–2024. Browse real held-out station-hours below, or "
            "nudge the current reading to see how the forecast responds."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    stations = sorted(df["Station"].unique())

    left, right = st.columns(
        [1, 1.15],
        gap="large",
    )

    # -----------------------------------------------------------------------
    # Station / timestamp selection
    # -----------------------------------------------------------------------

    with left:
        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="card-label">Select a station-hour</div>',
            unsafe_allow_html=True,
        )

        station = st.selectbox(
            "Station",
            stations,
            label_visibility="collapsed",
        )

        station_df = df[df["Station"] == station].sort_values("Datetime").reset_index(drop=True)

        timestamps = station_df["Datetime"]

        # Keyed to the station so switching stations resets to hour 0
        # rather than carrying over an index that may not exist for the
        # newly selected station.
        slider_key = f"hour_idx__{station}"

        if slider_key not in st.session_state:
            st.session_state[slider_key] = 0

        col_a, col_b = st.columns([3, 1])

        with col_b:
            # Mutate session_state before the slider widget below is
            # instantiated this run. This ensures that the slider itself
            # receives the updated random value and stays synchronized
            # with the snapshot displayed by the app.
            if st.button(
                "🎲 Random",
                width="stretch",
            ):
                st.session_state[slider_key] = int(
                    np.random.randint(
                        0,
                        len(station_df),
                    )
                )

        with col_a:
            idx = st.select_slider(
                "Hour",
                options=list(range(len(station_df))),
                format_func=lambda index: timestamps.iloc[index].strftime("%d %b %Y, %H:%M"),
                key=slider_key,
                label_visibility="collapsed",
            )

        snapshot = station_df.iloc[idx]

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="card-label">Current-hour readings</div>',
            unsafe_allow_html=True,
        )

        units = {
            "PM2.5": "µg/m³",
            "NO2": "µg/m³",
            "PM10": "µg/m³",
            "O3": "µg/m³",
            "Temperature": "°C",
            "Humidity": "%",
            "WindSpeed": "kn",
        }

        for field in CURRENT_HOUR_INPUTS:
            st.markdown(
                (
                    f'<div class="field-row">'
                    f"<span>{field}</span>"
                    f"<span>{snapshot[field]:.1f} "
                    f"{units.get(field, '')}</span>"
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Baseline prediction
    # -----------------------------------------------------------------------

    baseline_prediction = predict_row(
        model=model,
        imputer=imputer,
        row=snapshot,
        features=features,
    )

    actual = float(snapshot["PM2.5_target"])

    with right:
        st.markdown(
            '<div class="glass-card readout-wrap">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="card-label">Forecast · next hour</div>',
            unsafe_allow_html=True,
        )

        band_label, band_color = daqi_band(baseline_prediction)

        st.markdown(
            (
                f'<span class="readout-value">'
                f"{baseline_prediction:.1f}"
                f"</span>"
                f'<span class="readout-unit">'
                f"µg/m³ PM2.5"
                f"</span><br/>"
                f'<span class="daqi-tag" '
                f'style="background:{band_color};">'
                f"{band_label} · UK DAQI-style"
                f"</span>"
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="card-label">What actually happened</div>',
            unsafe_allow_html=True,
        )

        error = actual - baseline_prediction

        delta_class = "delta-good" if abs(error) <= 2 else "delta-bad"

        st.markdown(
            (
                f'<span class="readout-value" '
                f'style="font-size:2.2rem;">'
                f"{actual:.1f}"
                f"</span>"
                f'<span class="readout-unit">'
                f"µg/m³ actual"
                f"</span><br/>"
                f'<span class="{delta_class}">'
                f"Δ {error:+.2f} µg/m³ vs. forecast"
                f"</span>"
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<br/>",
        unsafe_allow_html=True,
    )

    render_what_if(
        snapshot=snapshot,
        model=model,
        imputer=imputer,
        features=features,
        baseline_prediction=baseline_prediction,
    )

    render_model_diagnostics(reports)


if __name__ == "__main__":
    main()
