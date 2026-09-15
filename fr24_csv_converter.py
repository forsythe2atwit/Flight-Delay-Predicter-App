from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

PARAMS = [
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation_probability",
    "rain",
    "showers",
    "snowfall",
    "snow_depth",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_direction_120m",
    "wind_direction_180m",
    "wind_gusts_10m",
    "temperature_80m",
    "temperature_120m",
    "temperature_180m",
    "visibility",
]

# Rename current Open-Meteo names back to the names expected by
# Weather_history_Predictor.py.
OUTPUT_NAME_MAP = {
    "wind_speed_10m": "windspeed_10m",
    "wind_speed_80m": "windspeed_80m",
    "wind_speed_120m": "windspeed_120m",
    "wind_speed_180m": "windspeed_180m",
    "wind_direction_10m": "winddirection_10m",
    "wind_direction_80m": "winddirection_80m",
    "wind_direction_120m": "winddirection_120m",
    "wind_direction_180m": "winddirection_180m",
    "wind_gusts_10m": "windgusts_10m",
}


def _parse_position(value: str) -> tuple[float, float]:
    parts = str(value).split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid Position value: {value!r}")
    return float(parts[0].strip()), float(parts[1].strip())


def _parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _open_meteo_endpoint(dt: datetime) -> str:
    # Archive endpoint is appropriate for historical FR24 tracks.
    # Forecast endpoint is retained for very recent/current tracks.
    age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    if age_days > 5:
        return "https://archive-api.open-meteo.com/v1/archive"
    return "https://api.open-meteo.com/v1/forecast"


def convert_fr24_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    points: int = 50,
    request_timeout: int = 30,
) -> Path:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)

    required = {"Position", "UTC"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{input_csv.name} is missing required FR24 column(s): "
            + ", ".join(sorted(missing))
        )

    df = df.dropna(subset=["Position", "UTC"]).copy()
    if df.empty:
        raise ValueError(f"{input_csv.name} contains no usable Position/UTC rows.")

    # Keep the number of samples <= number of available rows.
    n_points = min(max(1, points), len(df))
    sample_indices = np.linspace(0, len(df) - 1, n_points, dtype=int)
    sampled = df.iloc[sample_indices].reset_index(drop=True)

    session = requests.Session()
    results: list[dict] = []

    print(f"Sampling {n_points} points from {input_csv.name}...")

    for i, row in sampled.iterrows():
        lat, lon = _parse_position(row["Position"])
        dt = _parse_utc(row["UTC"])
        date_str = dt.strftime("%Y-%m-%d")
        hour_str = dt.strftime("%Y-%m-%dT%H:00")
        endpoint = _open_meteo_endpoint(dt)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": ",".join(PARAMS),
            "timezone": "UTC",
        }

        print(f"[{i + 1:02d}/{n_points}] Weather for {lat:.4f}, {lon:.4f} at {hour_str} UTC")

        last_error = None
        for attempt in range(3):
            try:
                response = session.get(endpoint, params=params, timeout=request_timeout)
                response.raise_for_status()
                data = response.json()
                hourly = data.get("hourly")
                if not hourly or "time" not in hourly:
                    raise RuntimeError(f"Open-Meteo returned no hourly data: {data}")

                if hour_str not in hourly["time"]:
                    raise RuntimeError(f"Requested hour {hour_str} not present in response.")

                idx = hourly["time"].index(hour_str)
                weather_row = {
                    "id": f"point{i + 1}",
                    "lat": lat,
                    "lon": lon,
                    "time": hour_str,
                }

                for api_name in PARAMS:
                    output_name = OUTPUT_NAME_MAP.get(api_name, api_name)
                    values = hourly.get(api_name)
                    weather_row[output_name] = (
                        values[idx] if values is not None and idx < len(values) else None
                    )

                results.append(weather_row)
                last_error = None
                break

            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))

        if last_error is not None:
            print(f"  WARNING: skipped point {i + 1}: {last_error}")

    if not results:
        raise RuntimeError(
            "No weather points were returned. Check your internet connection, "
            "FR24 timestamps, and Open-Meteo availability."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)

    print(f"Saved {len(results)} weather points -> {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an FR24 track CSV to Open-Meteo weather history.")
    parser.add_argument("--in", dest="input_csv", required=True, help="FR24 CSV path")
    parser.add_argument("--out", dest="output_csv", required=True, help="Weather CSV output path")
    parser.add_argument("--points", type=int, default=50, help="Number of trajectory samples (default: 50)")
    args = parser.parse_args()

    convert_fr24_csv(args.input_csv, args.output_csv, points=args.points)


if __name__ == "__main__":
    main()
