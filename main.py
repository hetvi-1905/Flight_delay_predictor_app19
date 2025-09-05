# main.py
import pandas as pd
from models.forecast_model import forecast_next
from models.attribution_model import load_or_train, predict_breakdown
from models.llm_explainer import generate_explanation

# Load dataset
hist = pd.read_csv("data/flights_2025.csv")

# User input
carrier_code = "9E"
airport_code = "ABE"
target_month = "2025-05"

# Filter carrier + airport
subset = hist[(hist["carrier"] == carrier_code) & (hist["airport"] == airport_code)].copy()

# Forecast
avg_delay = forecast_next(subset, target_month)
print(f"⏳ Avg Delay: {avg_delay} minutes per flight")

# Attribution
aobj = load_or_train(subset, carrier=carrier_code, airport=airport_code)
row = subset.dropna(subset=aobj['features']).iloc[-1]
breakdown = predict_breakdown(aobj, row)
print("📊 Breakdown:", breakdown)

# Fetch names (if available)
carrier_name = subset["carrier_name"].iloc[-1] if "carrier_name" in subset else carrier_code
airport_name = subset["airport_name"].iloc[-1] if "airport_name" in subset else airport_code

# LLM Explanation
explanation = generate_explanation(carrier_name, airport_name, target_month, avg_delay, breakdown)
print("📝 Explanation:", explanation)
