"""Shared visual styling for the Streamlit dashboard."""

import base64
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
BACKGROUND_IMAGE = APP_DIR / "assets" / "london_skyline.jpg"


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

        .stApp {{
            background-image:
                linear-gradient(
                    180deg,
                    rgba(8,10,16,0.55) 0%,
                    rgba(8,10,16,0.75) 45%,
                    rgba(6,8,13,0.94) 100%
                ),
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
            color: #C4BBA8;
            margin-bottom: 0.9rem;
        }}

        .readout-wrap {{
            animation: rise 0.5s ease-out;
        }}

        @keyframes rise {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
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
            color: #C4BBA8;
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
            color: #F0EAD9;
            padding: 0.35rem 0;
            border-bottom: 1px dashed rgba(255,255,255,0.08);
        }}

        .field-row span:first-child {{
            color: #C4BBA8;
        }}

        .delta-good {{
            color: #7FD196;
            font-family: 'IBM Plex Mono', monospace;
        }}

        .delta-bad {{
            color: #F0A183;
            font-family: 'IBM Plex Mono', monospace;
        }}

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
            color: #A69C89;
            text-align: center;
            margin-top: 3rem;
            letter-spacing: 0.05em;
        }}

        /*
        Native Streamlit text — tabs, expander, captions, metrics,
        dataframe, and body markdown — had no color override before,
        so it rendered in Streamlit's default dark text directly on
        top of the dark photo background. Forcing a light, readable
        color here fixes that across the whole app.
        */

        .stMarkdown,
        .stMarkdown p,
        .stMarkdown li,
        .stMarkdown strong {{
            color: #F0EAD9;
        }}

        .stCaption,
        [data-testid="stCaptionContainer"] {{
            color: #C4BBA8 !important;
        }}

        .stTabs [data-baseweb="tab"] p {{
            color: #E4DECF !important;
            font-weight: 500;
        }}

        .stTabs [aria-selected="true"] p {{
            color: #F4EFE4 !important;
        }}

        [data-testid="stExpander"] summary p {{
            color: #F0EAD9 !important;
        }}

        [data-testid="stMetricLabel"] {{
            color: #C4BBA8 !important;
        }}

        [data-testid="stMetricValue"] {{
            color: #F4EFE4 !important;
        }}

        [data-testid="stMetricDelta"] {{
            color: #E4DECF !important;
        }}

        label,
        .stSelectbox label {{
            color: #E4DECF !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
