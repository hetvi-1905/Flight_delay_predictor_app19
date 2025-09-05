import pandas as pd

# Load dataset
# df = pd.read_csv("C:\Users\Admin\Desktop\____LTI project\Project 2 Flight wala dataset\data\flights_2025.csv")
df = pd.read_csv("C:/Users/Admin/Desktop/____LTI project/Project 2 Flight wala dataset/data/flights_2025.csv")

# Create a proper datetime column for grouping
df["month"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")

# Rename arr_delay → avg_delay (what forecast_model expects)
df = df.rename(columns={"arr_delay": "avg_delay"})

# Keep only the relevant subset of columns
df = df[[
    "month", "carrier", "airport", "avg_delay",
    "carrier_delay", "weather_delay", "nas_delay", 
    "security_delay", "late_aircraft_delay"
]]

df.to_csv(r"C:\Users\Admin\Desktop\____LTI project\Project 2 Flight wala dataset\data\flights_2025.csv", index=False)

