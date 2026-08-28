"""Model diagnostics and project-information UI for Streamlit."""

import plotly.graph_objects as go
import streamlit as st


def render_model_diagnostics(reports):
    # -----------------------------------------------------------------------
    # Model diagnostics
    # -----------------------------------------------------------------------

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Feature importance",
            "📍 Accuracy by station",
            "ℹ️ About this model",
        ]
    )

    with tab1:
        fi = reports["feature_importance.csv"]

        if fi is not None:
            top = fi.head(12).sort_values("Importance")

            fig = go.Figure(
                go.Bar(
                    x=top["Importance"],
                    y=top["Feature"],
                    orientation="h",
                    marker_color="#D9A544",
                )
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={
                    "family": "Inter",
                    "color": "#F4EFE4",
                },
                height=420,
                margin={
                    "l": 10,
                    "r": 10,
                    "t": 10,
                    "b": 10,
                },
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

        else:
            st.info("Run `python -m src.evaluate` to generate feature importance.")

    with tab2:
        se = reports["station_level_errors.csv"]

        if se is not None:
            se = se.sort_values("RMSE")

            fig = go.Figure(
                go.Bar(
                    x=se["RMSE"],
                    y=se[se.columns[0]],
                    orientation="h",
                    marker_color="#C15B2C",
                )
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={
                    "family": "Inter",
                    "color": "#F4EFE4",
                },
                height=340,
                margin={
                    "l": 10,
                    "r": 10,
                    "t": 10,
                    "b": 10,
                },
                xaxis_title="RMSE (µg/m³)",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            st.caption(
                "Marylebone Road is a roadside traffic-monitoring "
                "site, so PM2.5 there is noisier and harder to "
                "forecast than the background/suburban stations."
            )

        else:
            st.info("Run `python -m src.evaluate` to generate station-level error breakdowns.")

    with tab3:
        results = reports["baseline_model_results.csv"]

        st.markdown(
            "**Model**: Random Forest, "
            "`n_estimators=100, random_state=42`, "
            "trained on 2021–2023, evaluated on "
            "held-out 2024."
        )

        if results is not None:
            st.dataframe(
                results.set_index("Model").style.format("{:.3f}"),
                width="stretch",
            )

        st.markdown(
            "**Target construction**: PM2.5 one hour ahead, "
            "matched on `(Station, Datetime + 1h)` rather than "
            "a plain row shift — the raw data has timestamp gaps, "
            "and shifting rows across a gap silently produces the "
            "wrong target."
        )

        st.markdown(
            "**Data**: DEFRA UK-AIR AURN network (pollutants) + "
            "Met Office MIDAS Heathrow (weather), 2021–2024."
        )

    st.markdown(
        (
            '<div class="footer-note">'
            "RANDOM FOREST · SCIKIT-LEARN · STREAMLIT · "
            "DATA: DEFRA AURN &amp; MET OFFICE MIDAS"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
