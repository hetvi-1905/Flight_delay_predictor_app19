# 


# app.py
import streamlit as st
import pandas as pd
import datetime
from models.forecast_model import forecast_next
from models.attribution_model import load_or_train, predict_breakdown
from models.llm_explainer import generate_explanation

st.title("✈️ Flight Delay Predictor & Analyzer (Future Forecast)")

# --- LOAD HISTORICAL DATA ---
hist = pd.read_csv("data/flights_2025.csv")

# --- USER INPUTS ---
# Carrier dropdown
carrier_options = hist["carrier"].unique().tolist()
carrier_code = st.selectbox("Select Carrier", carrier_options)

# Airport dropdown filtered by selected carrier
airport_options = hist[hist["carrier"] == carrier_code]["airport"].unique().tolist()
airport_code = st.selectbox("Select Airport", airport_options)

# Future month picker (only future months)
today = datetime.date.today()
# default to next month
default_month = (today.replace(day=1) + datetime.timedelta(days=31)).replace(day=1)
target_month_date = st.date_input(
    "Select Month for Forecast",
    value=default_month,
    min_value=today
)
target_month = target_month_date.strftime("%Y-%m")  # convert to YYYY-MM

# --- FILTER HISTORICAL DATA FOR TRAINING ---
subset = hist[(hist["carrier"] == carrier_code) & (hist["airport"] == airport_code)].copy()

if subset.empty:
    st.warning("No historical data available for the selected carrier and airport.")
else:
    # --- FORECAST ---
    avg_delay = forecast_next(subset, target_month)
    st.subheader("⏳ Average Delay (Forecast)")
    st.write(f"{avg_delay:.2f} minutes per flight")

    # --- ATTRIBUTION / BREAKDOWN ---
    aobj = load_or_train(subset, carrier=carrier_code, airport=airport_code)
    # Use the latest historical row to generate breakdown for future forecast
    row = subset.dropna(subset=aobj['features']).iloc[-1]
    breakdown = predict_breakdown(aobj, row)
    st.subheader("📊 Delay Breakdown (Estimated Contributors)")
    st.json(breakdown)

    # --- FETCH NAMES ---
    carrier_name = subset["carrier_name"].iloc[-1] if "carrier_name" in subset else carrier_code
    airport_name = subset["airport_name"].iloc[-1] if "airport_name" in subset else airport_code

    # --- LLM EXPLANATION ---
    st.subheader("📝 Explanation")
    with st.spinner("Generating explanation..."):
        explanation = generate_explanation(carrier_name, airport_name, target_month, avg_delay, breakdown)
        st.write(explanation)


#  streamlit run app.py
