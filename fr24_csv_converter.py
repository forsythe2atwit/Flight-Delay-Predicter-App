import requests
import pandas as pd
import numpy as np
from datetime import datetime

CSV_FILE = "flights_csv/MIA_BOS.csv"
OUTPUT_CSV = "openmeteo_flight_weather_history.csv"

PARAMS = [
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation_probability",
    "rain",
    "showers",
    "snowfall",
    "snow_depth",
    "windspeed_10m", "windspeed_80m", "windspeed_120m", "windspeed_180m",
    "winddirection_10m", "winddirection_80m", "winddirection_120m", "winddirection_180m",
    "windgusts_10m",
    "temperature_80m", "temperature_120m", "temperature_180m",
    "visibility"
]

df = pd.read_csv(CSV_FILE)
positions = df["Position"].tolist()
times = df["UTC"].tolist()

sample_indices = np.linspace(0, len(df) - 1, 50, dtype=int)
sampled_positions = [positions[i] for i in sample_indices]
sampled_times = [times[i] for i in sample_indices]

results = []
for i, (pos, tstamp) in enumerate(zip(sampled_positions, sampled_times)):
    lat, lon = pos.split(",")

    dt = datetime.fromisoformat(tstamp.replace("Z", "+00:00"))
    date_str = dt.strftime("%Y-%m-%d")
    hour_str = dt.strftime("%Y-%m-%dT%H:00")

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&hourly={','.join(PARAMS)}"
        f"&timezone=UTC"
    )

    print(f"Fetching {url}")
    r = requests.get(url)
    data = r.json()
    if "hourly" not in data:
        continue

    hourly = data["hourly"]
    if hour_str in hourly["time"]:
        idx = hourly["time"].index(hour_str)
        row = {"id": f"point{i+1}", "lat": float(lat), "lon": float(lon), "time": hour_str}
        for param in PARAMS:
            row[param] = hourly.get(param, [None])[idx]
        results.append(row)

df_weather = pd.DataFrame(results)
df_weather.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved historical weather results to {OUTPUT_CSV}")
print(df_weather.head())
