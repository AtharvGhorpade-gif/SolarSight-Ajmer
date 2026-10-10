
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Ajmer Solar Predictor",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Ajmer Solar Irradiation Predictor")

st.write(
    "A machine-learning prototype that predicts approximate "
    "daily solar irradiation using historical observations."
)

BASE = Path(__file__).parent

DATA_FILE = BASE / "ajmer_daily_solar_irradiation.csv"
MODEL_FILE = BASE / "ajmer_solar_model.joblib"
METRICS_FILE = BASE / "ajmer_model_metrics.json"

# Check that all required files exist
if not all(
    path.exists()
    for path in [DATA_FILE, MODEL_FILE, METRICS_FILE]
):
    st.error(
        "Required files are missing. Keep the daily CSV, trained "
        "model, and metrics JSON in the same folder as app.py."
    )
    st.stop()

# Load the saved project files
daily = pd.read_csv(
    DATA_FILE,
    index_col="Date",
    parse_dates=True
)

model = joblib.load(MODEL_FILE)

with open(METRICS_FILE, "r") as f:
    metrics = json.load(f)

history = (
    daily["Solar_Irradiation_kWh_m2"]
    .dropna()
    .sort_index()
)

# Display model performance
st.subheader("Model Evaluation")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Random Forest MAE",
    f'{metrics["random_forest"]["MAE"]:.3f}'
)

c2.metric(
    "Random Forest RMSE",
    f'{metrics["random_forest"]["RMSE"]:.3f}'
)

c3.metric(
    "Random Forest R²",
    f'{metrics["random_forest"]["R2"]:.3f}'
)

st.caption(
    "These metrics are calculated on the held-out test period. "
    "Lower MAE and RMSE are better."
)

# User inputs
st.subheader("Next-Day Solar Estimate")

capacity = st.number_input(
    "Solar installation capacity (kWp)",
    min_value=0.1,
    max_value=10000.0,
    value=1.0,
    step=0.5
)

performance_ratio = st.slider(
    "Assumed system performance ratio",
    min_value=0.50,
    max_value=0.95,
    value=0.80,
    step=0.01
)

if len(history) < 14:
    st.error("At least 14 valid daily records are required.")
    st.stop()

# Forecast the day after the last available observation
last_date = history.index.max()
forecast_date = last_date + pd.Timedelta(days=1)

features = {
    "day_of_year": forecast_date.dayofyear,
    "month": forecast_date.month,
    "day_of_week": forecast_date.dayofweek,
    "lag_1": history.iloc[-1],
    "lag_2": history.iloc[-2],
    "lag_3": history.iloc[-3],
    "lag_7": history.iloc[-7],
    "lag_14": history.iloc[-14]
}

X_next = pd.DataFrame([features])

predicted_irradiation = max(
    0.0,
    float(model.predict(X_next)[0])
)

# Illustrative PV energy estimate
estimated_energy = (
    predicted_irradiation
    * capacity
    * performance_ratio
)

st.write(
    "Forecast date:",
    forecast_date.strftime("%d %B %Y")
)

left, right = st.columns(2)

left.metric(
    "Estimated Daily Irradiation",
    f"{predicted_irradiation:.2f} kWh/m²/day"
)

right.metric(
    "Estimated PV Energy",
    f"{estimated_energy:.2f} kWh/day"
)

st.caption(
    "This forecast uses historical patterns, not a live weather "
    "forecast. PV energy is an estimate based on the selected "
    "capacity and assumed performance ratio."
)

# Historical chart
st.subheader("Recent Solar Irradiation")

recent = history.tail(180)

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(recent.index, recent.values)

ax.set_xlabel("Date")
ax.set_ylabel("Approx. irradiation (kWh/m²/day)")
ax.grid(True, alpha=0.3)

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Location and limitations
st.subheader("Research Location")

st.write(
    "Target coordinates: 26.250000° N, 74.666667° E, Ajmer, "
    "Rajasthan. The observations come from the selected station, "
    "which may be some distance from the target coordinates."
)

st.info(
    "This is an educational prototype. Daily irradiation is "
    "estimated from hourly radiation readings. Missing readings "
    "can affect accuracy, and estimated PV energy is not measured "
    "electricity generation."
)
