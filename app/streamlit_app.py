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

import base64
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "london_air_quality_features_2021_2024.csv"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "baseline_model_results.csv"
BACKGROUND_IMAGE = APP_DIR / "assets" / "london_skyline.jpg"

CURRENT_HOUR_INPUTS = [
    "PM2.5", "NO2", "PM10", "O3",
    "Temperature", "Humidity", "WindSpeed",
]

# UK DEFRA Daily Air Quality Index style bands for PM2.5 (µg/m3).
# DAQI is officially a 24-hour mean; applying its thresholds to a single
# hourly forecast is an approximation, made explicit in the UI.
DAQI_BANDS = [
    (0, 11, "Low", "#4E8B63"),
    (11, 23, "Moderate", "#C9A227"),
    (23, 35, "High", "#C15B2C"),
    (35, float("inf"), "Very High", "#8C2F39"),
]


def daqi_band(value: float):
    for low, high, label, color in DAQI_BANDS:
        if low <= value < high:
            return label, color
    return DAQI_BANDS[-1][2], DAQI_BANDS[-1][3]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

@st.cache_data
def _background_b64() -> str:
    return base64.b64encode(BACKGROUND_IMAGE.read_bytes()).decode()


def inject_style():
    bg = _background_b64()
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        #MainMenu, footer, header {{ visibility: hidden; }}

        .stApp {{
            background-image:
                linear-gradient(180deg, rgba(8,10,16,0.55) 0%, rgba(8,10,16,0.75) 45%, rgba(6,8,13,0.94) 100%),
                url("data:image/jpeg;base64,{bg}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            max-width: 1100px;
            padding-top: 2.5rem;
        }}

        .eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: #D9A544;
            margin-bottom: 0.4rem;
        }}

        .hero-title {{
            font-family: 'Playfair Display', serif;
            font-weight: 700;
            font-size: 3rem;
            line-height: 1.05;
            color: #F4EFE4;
            margin-bottom: 0.6rem;
            text-shadow: 0 2px 24px rgba(0,0,0,0.45);
        }}

        .hero-sub {{
            font-size: 1.02rem;
            color: #C9BFA8;
            max-width: 640px;
            line-height: 1.5;
            margin-bottom: 2.2rem;
        }}

        .glass-card {{
            background: rgba(14, 16, 22, 0.58);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 14px;
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.2rem;
        }}

        .card-label {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #9C9484;
            margin-bottom: 0.9rem;
        }}

        .readout-wrap {{
            animation: rise 0.5s ease-out;
        }}
        @keyframes rise {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .readout-value {{
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 600;
            font-size: 3.6rem;
            color: #F4EFE4;
            letter-spacing: 0.02em;
            line-height: 1;
            border-bottom: 2px solid rgba(255,255,255,0.12);
            display: inline-block;
            padding-bottom: 0.3rem;
        }}
        .readout-unit {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1rem;
            color: #9C9484;
            margin-left: 0.5rem;
        }}

        .daqi-tag {{
            display: inline-block;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            padding: 0.3rem 0.7rem;
            border-radius: 100px;
            margin-top: 0.9rem;
            color: #0B0C10;
            font-weight: 600;
        }}

        .field-row {{
            display: flex;
            justify-content: space-between;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.88rem;
            color: #E4DECF;
            padding: 0.35rem 0;
            border-bottom: 1px dashed rgba(255,255,255,0.08);
        }}
        .field-row span:first-child {{ color: #9C9484; }}

        .delta-good {{ color: #6FBE87; font-family: 'IBM Plex Mono', monospace; }}
        .delta-bad {{ color: #E08A6B; font-family: 'IBM Plex Mono', monospace; }}

        section[data-testid="stSidebar"] {{
            background: rgba(10, 12, 18, 0.75);
            backdrop-filter: blur(14px);
        }}

        div[data-testid="stMetric"] {{
            background: rgba(14, 16, 22, 0.5);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 0.8rem 1rem;
        }}

        .footer-note {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: #786F5E;
            text-align: center;
            margin-top: 3rem;
            letter-spacing: 0.05em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_bundle():
    model = joblib.load(MODEL_DIR / "random_forest_pm25.pkl")
    imputer = joblib.load(MODEL_DIR / "imputer.pkl")
    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    return model, imputer, metadata


@st.cache_data
def load_test_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
    df = df[df["Datetime"].dt.year == 2024].reset_index(drop=True)
    return df


@st.cache_data
def load_reports():
    reports = {}
    for name in ["feature_importance.csv", "station_level_errors.csv"]:
        path = REPORTS_DIR / name
        reports[name] = pd.read_csv(path) if path.exists() else None
    reports["baseline_model_results.csv"] = (
        pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else None
    )
    return reports


def predict_row(model, imputer, features, row: pd.Series) -> float:
    X = pd.DataFrame([row[features].values], columns=features)
    X_imputed = pd.DataFrame(imputer.transform(X), columns=features)
    return float(model.predict(X_imputed)[0])


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

    missing = [
        p for p in [BACKGROUND_IMAGE, MODEL_DIR / "random_forest_pm25.pkl",
                    MODEL_DIR / "imputer.pkl", MODEL_DIR / "model_metadata.json", DATA_PATH]
        if not p.exists()
    ]
    if missing:
        st.error(
            "Missing required files:\n\n"
            + "\n".join(f"- `{p.relative_to(PROJECT_ROOT)}`" for p in missing)
            + "\n\nRun the pipeline first: `python -m src.data_processing` → "
              "`python -m src.feature_engineering` → `python -m src.train`."
        )
        return

    model, imputer, metadata = load_model_bundle()
    features = metadata["features"]
    df = load_test_data()
    reports = load_reports()

    st.markdown('<div class="eyebrow">LONDON · FIVE MONITORING STATIONS · HELD-OUT 2024</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Air Quality Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">One-hour-ahead PM2.5 forecasts from a Random Forest trained on '
        'AURN pollution monitoring and Heathrow weather data, 2021–2024. '
        'Browse real held-out station-hours below, or nudge the current reading to see how the forecast responds.</div>',
        unsafe_allow_html=True,
    )

    stations = sorted(df["Station"].unique())

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Select a station-hour</div>', unsafe_allow_html=True)

        station = st.selectbox("Station", stations, label_visibility="collapsed")
        station_df = df[df["Station"] == station].sort_values("Datetime").reset_index(drop=True)
        timestamps = station_df["Datetime"]

        # Keyed to the station so switching stations resets to hour 0 rather than
        # carrying over an index that may not exist for the new station.
        slider_key = f"hour_idx__{station}"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = 0

        col_a, col_b = st.columns([3, 1])
        with col_b:
            # Mutate session_state *before* the slider widget below is instantiated
            # this run, so the slider itself picks up the new value — rather than
            # overriding a local variable after the fact, which would desync the
            # slider's displayed position from the snapshot actually shown.
            if st.button("🎲 Random", use_container_width=True):
                st.session_state[slider_key] = int(np.random.randint(0, len(station_df)))

        with col_a:
            idx = st.select_slider(
                "Hour",
                options=list(range(len(station_df))),
                format_func=lambda i: timestamps.iloc[i].strftime("%d %b %Y, %H:%M"),
                key=slider_key,
                label_visibility="collapsed",
            )

        snapshot = station_df.iloc[idx]
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Current-hour readings</div>', unsafe_allow_html=True)
        units = {
            "PM2.5": "µg/m³", "NO2": "µg/m³", "PM10": "µg/m³", "O3": "µg/m³",
            "Temperature": "°C", "Humidity": "%", "WindSpeed": "kn",
        }
        for field in CURRENT_HOUR_INPUTS:
            st.markdown(
                f'<div class="field-row"><span>{field}</span>'
                f'<span>{snapshot[field]:.1f} {units.get(field, "")}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    baseline_prediction = predict_row(model, imputer, features, snapshot)
    actual = float(snapshot["PM2.5_target"])

    with right:
        st.markdown('<div class="glass-card readout-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Forecast · next hour</div>', unsafe_allow_html=True)
        band_label, band_color = daqi_band(baseline_prediction)
        st.markdown(
            f'<span class="readout-value">{baseline_prediction:.1f}</span>'
            f'<span class="readout-unit">µg/m³ PM2.5</span><br/>'
            f'<span class="daqi-tag" style="background:{band_color};">{band_label} · UK DAQI-style</span>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">What actually happened</div>', unsafe_allow_html=True)
        error = actual - baseline_prediction
        delta_class = "delta-good" if abs(error) <= 2 else "delta-bad"
        st.markdown(
            f'<span class="readout-value" style="font-size:2.2rem;">{actual:.1f}</span>'
            f'<span class="readout-unit">µg/m³ actual</span><br/>'
            f'<span class="{delta_class}">Δ {error:+.2f} µg/m³ vs. forecast</span>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    with st.expander("🔧 What-if — nudge the current reading"):
        st.caption(
            "Adjusts only this hour's observed readings. Recent history (lags, rolling "
            "averages) stays fixed at the real values above, so the forecast shift you see "
            "reflects just the change you made — not a fabricated history."
        )
        wcol1, wcol2 = st.columns(2)
        overrides = {}
        with wcol1:
            overrides["PM2.5"] = st.slider("Current PM2.5 (µg/m³)", 0.0, 120.0, float(snapshot["PM2.5"]), 0.5)
            overrides["NO2"] = st.slider("NO2 (µg/m³)", 0.0, 200.0, float(snapshot["NO2"]), 1.0)
            overrides["Temperature"] = st.slider("Temperature (°C)", -10.0, 40.0, float(snapshot["Temperature"]), 0.5)
        with wcol2:
            overrides["PM10"] = st.slider("PM10 (µg/m³)", 0.0, 200.0, float(snapshot["PM10"]), 1.0)
            overrides["Humidity"] = st.slider("Humidity (%)", 0.0, 100.0, float(snapshot["Humidity"]), 1.0)
            overrides["WindSpeed"] = st.slider("Wind speed (kn)", 0.0, 60.0, float(snapshot["WindSpeed"]), 0.5)

        whatif_row = snapshot.copy()
        for field, value in overrides.items():
            whatif_row[field] = value
        whatif_prediction = predict_row(model, imputer, features, whatif_row)
        shift = whatif_prediction - baseline_prediction

        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Baseline forecast", f"{baseline_prediction:.1f} µg/m³")
        wc2.metric("What-if forecast", f"{whatif_prediction:.1f} µg/m³", f"{shift:+.2f}")
        band_label_w, _ = daqi_band(whatif_prediction)
        wc3.metric("DAQI-style band", band_label_w)

    tab1, tab2, tab3 = st.tabs(["📊 Feature importance", "📍 Accuracy by station", "ℹ️ About this model"])

    with tab1:
        fi = reports["feature_importance.csv"]
        if fi is not None:
            top = fi.head(12).sort_values("Importance")
            fig = go.Figure(go.Bar(
                x=top["Importance"], y=top["Feature"], orientation="h",
                marker_color="#D9A544",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#E4DECF"), height=420,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run `python -m src.evaluate` to generate feature importance.")

    with tab2:
        se = reports["station_level_errors.csv"]
        if se is not None:
            se = se.sort_values("RMSE")
            fig = go.Figure(go.Bar(
                x=se["RMSE"], y=se[se.columns[0]], orientation="h",
                marker_color="#C15B2C",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#E4DECF"), height=340,
                margin=dict(l=10, r=10, t=10, b=10), xaxis_title="RMSE (µg/m³)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Marylebone Road is a roadside traffic-monitoring site, so PM2.5 there is "
                "noisier and harder to forecast than the background/suburban stations."
            )
        else:
            st.info("Run `python -m src.evaluate` to generate station-level error breakdowns.")

    with tab3:
        results = reports["baseline_model_results.csv"]
        st.markdown("**Model**: Random Forest, `n_estimators=100, random_state=42`, trained on 2021–2023, evaluated on held-out 2024.")
        if results is not None:
            st.dataframe(results.set_index("Model").style.format("{:.3f}"), use_container_width=True)
        st.markdown(
            "**Target construction**: PM2.5 one hour ahead, matched on "
            "`(Station, Datetime + 1h)` rather than a plain row shift — the raw data has "
            "timestamp gaps, and shifting rows across a gap silently produces the wrong target."
        )
        st.markdown(
            "**Data**: DEFRA UK-AIR AURN network (pollutants) + Met Office MIDAS Heathrow "
            "(weather), 2021–2024."
        )

    st.markdown(
        '<div class="footer-note">RANDOM FOREST · SCIKIT-LEARN · STREAMLIT · '
        'DATA: DEFRA AURN &amp; MET OFFICE MIDAS</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
