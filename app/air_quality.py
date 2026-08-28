"""Air-quality presentation helpers shared across the Streamlit dashboard."""

DAQI_BANDS = [
    (0, 11, "Low", "#4E8B63"),
    (11, 23, "Moderate", "#C9A227"),
    (23, 35, "High", "#C15B2C"),
    (35, float("inf"), "Very High", "#8C2F39"),
]


def daqi_band(value: float):
    """Return the approximate PM2.5 DAQI-style label and display colour."""
    for low, high, label, color in DAQI_BANDS:
        if low <= value < high:
            return label, color

    return DAQI_BANDS[-1][2], DAQI_BANDS[-1][3]
