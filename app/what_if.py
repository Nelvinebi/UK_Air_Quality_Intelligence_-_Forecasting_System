"""Interactive What-if controls for the Streamlit dashboard."""

import streamlit as st

from app.air_quality import daqi_band
from src.inference import predict_row


def render_what_if(
    snapshot,
    model,
    imputer,
    features,
    baseline_prediction,
):
    """Render controls that vary current-hour readings and recompute the forecast."""

    with st.expander("🔧 What-if — nudge the current reading"):
        st.caption(
            "Adjusts only this hour's observed readings. "
            "Recent history (lags, rolling averages) stays fixed "
            "at the real values above, so the forecast shift you "
            "see reflects the current-hour change rather than a "
            "fabricated historical sequence."
        )

        wcol1, wcol2 = st.columns(2)

        overrides = {}

        with wcol1:
            overrides["PM2.5"] = st.slider(
                "Current PM2.5 (µg/m³)",
                0.0,
                120.0,
                float(snapshot["PM2.5"]),
                0.5,
            )

            overrides["NO2"] = st.slider(
                "NO2 (µg/m³)",
                0.0,
                200.0,
                float(snapshot["NO2"]),
                1.0,
            )

            overrides["Temperature"] = st.slider(
                "Temperature (°C)",
                -10.0,
                40.0,
                float(snapshot["Temperature"]),
                0.5,
            )

        with wcol2:
            overrides["PM10"] = st.slider(
                "PM10 (µg/m³)",
                0.0,
                200.0,
                float(snapshot["PM10"]),
                1.0,
            )

            overrides["Humidity"] = st.slider(
                "Humidity (%)",
                0.0,
                100.0,
                float(snapshot["Humidity"]),
                1.0,
            )

            overrides["WindSpeed"] = st.slider(
                "Wind speed (kn)",
                0.0,
                60.0,
                float(snapshot["WindSpeed"]),
                0.5,
            )

        whatif_row = snapshot.copy()

        for field, value in overrides.items():
            whatif_row[field] = value

        # Recompute engineered features that depend directly on the
        # current-hour variables modified by the What-if controls.
        #
        # Lag and rolling-history features deliberately remain fixed at
        # the genuine values associated with this station-hour.
        whatif_row["Temperature_Humidity"] = whatif_row["Temperature"] * whatif_row["Humidity"]

        whatif_row["WindSpeed_Squared"] = whatif_row["WindSpeed"] ** 2

        whatif_prediction = predict_row(
            model=model,
            imputer=imputer,
            row=whatif_row,
            features=features,
        )

        shift = whatif_prediction - baseline_prediction

        wc1, wc2, wc3 = st.columns(3)

        wc1.metric(
            "Baseline forecast",
            f"{baseline_prediction:.1f} µg/m³",
        )

        wc2.metric(
            "What-if forecast",
            f"{whatif_prediction:.1f} µg/m³",
            f"{shift:+.2f}",
        )

        band_label_w, _ = daqi_band(whatif_prediction)

        wc3.metric(
            "DAQI-style band",
            band_label_w,
        )
