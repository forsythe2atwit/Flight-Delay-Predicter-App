import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# Weather parameters we want
WEATHER_PARAMS = [
    "relative_humidity_2m","dew_point_2m","apparent_temperature",
    "precipitation_probability","rain","snowfall","snow_depth",
    "windspeed_10m","windgusts_10m","visibility"
]

def round_hour(ts):
    return pd.to_datetime(ts).floor("H")

def fetch_weather(lat, lon, date):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&hourly={','.join(WEATHER_PARAMS)}&timezone=UTC"
    )
    r = requests.get(url, timeout=15)
    return r.json()

def process_flight_csv(file_path, label_delay_min=None, sample_points=50):
    df = pd.read_csv(file_path)
    if "UTC" in df.columns:
        df["timestamp"] = pd.to_datetime(df["UTC"])
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("CSV must have UTC or timestamp column")

    # Ensure lat/lon exist
    if "lat" not in df.columns and "Position" in df.columns:
        df[["lat","lon"]] = df["Position"].str.split(",", expand=True).astype(float)

    df = df.sort_values("timestamp")

    # Sample ~50 points
    sample_idx = np.linspace(0, len(df) - 1, sample_points, dtype=int)
    sampled = df.iloc[sample_idx]

    records = []
    for _, row in sampled.iterrows():
        dt = round_hour(row["timestamp"])
        date_str = dt.strftime("%Y-%m-%d")
        data = fetch_weather(row["lat"], row["lon"], date_str)
        if "hourly" not in data:
            continue
        hour_str = dt.strftime("%Y-%m-%dT%H:00")
        if hour_str in data["hourly"]["time"]:
            idx = data["hourly"]["time"].index(hour_str)
            rec = {param: data["hourly"][param][idx] for param in WEATHER_PARAMS if param in data["hourly"]}
            records.append(rec)

    if not records:
        return None

    wx_df = pd.DataFrame(records)
    features = {
        "flight_file": os.path.basename(file_path),
        "max_precip_prob": wx_df["precipitation_probability"].max(),
        "min_visibility": wx_df["visibility"].min(),
        "max_windgusts": wx_df["windgusts_10m"].max(),
        "mean_temp": wx_df["apparent_temperature"].mean(),
        "dewpoint_spread": (wx_df["apparent_temperature"] - wx_df["dew_point_2m"]).min(),
        "mean_windspeed": wx_df["windspeed_10m"].mean(),
        "label_delay_min": label_delay_min
    }
    return features

def process_batch(folder, output_file="feature_matrix.csv"):
    all_feats = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            feats = process_flight_csv(os.path.join(folder, file))
            if feats:
                all_feats.append(feats)
    df = pd.DataFrame(all_feats)
    df.to_csv(output_file, index=False)
    print(f"✅ Saved feature matrix with {len(df)} flights → {output_file}")
    return df

# =============================
# Example usage
# =============================
if __name__ == "__main__":
    folder = "flights_csv"   # put all Flightradar24 CSVs here
    process_batch(folder, "batch_feature_matrix.csv")
