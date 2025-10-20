#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch runner:
- For every FR24 CSV in --in_dir, create an Open-Meteo flight weather history CSV (50 points)
- Then invoke your predictor script to produce *_per_point.csv and *_summary.csv

Usage:
  python batch_flight_pipeline.py --in_dir flights_csv --out_dir out --points 50
"""

import os, sys, glob, subprocess, time
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests

# ---------- Tunables ----------
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

POINTS_DEFAULT = 50
TIMEZONE = "UTC"  # keep UTC to match your risk script

from datetime import datetime, timezone
import pandas as pd

# --- NEW robust date/time parsing for FR24 "UTC" column ---
def _parse_utc(ts_str: str):
    if not isinstance(ts_str, str) or not ts_str.strip():
        return None

    # Try pandas' flexible parser first
    t = pd.to_datetime(ts_str, utc=True, errors="coerce")
    if pd.notna(t):
        return t

    # Try a few explicit formats common in FR24 exports
    fmts = [
        "%m/%d/%y %H:%M",
        "%m/%d/%y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(ts_str, fmt)
            return pd.Timestamp(dt, tz="UTC")
        except Exception:
            pass
    return None


def _iso_floor_to_hour(ts_str: str):
    """Return ISO string 'YYYY-MM-DDTHH:00' in UTC for Open-Meteo hourly match."""
    t = _parse_utc(ts_str)
    if t is None:
        return None
    t_round = t.round("H")
    return t_round.strftime("%Y-%m-%dT%H:00")


def _day_str(ts_str: str):
    """Return 'YYYY-MM-DD' (UTC) for Open-Meteo start/end_date params."""
    t = _parse_utc(ts_str)
    if t is None:
        return None
    return t.strftime("%Y-%m-%d")


def convert_one_fr24_csv(fr24_csv_path: Path, out_csv_path: Path, points: int = POINTS_DEFAULT) -> int:
    """
    Convert one FR24 CSV into an Open-Meteo flight weather history CSV with ~points rows.
    Returns number of points written.
    """
    df = pd.read_csv(fr24_csv_path)

    # Expect FR24 columns named like: "Position" and "UTC"
    if "Position" not in df.columns or "UTC" not in df.columns:
        raise ValueError(f"{fr24_csv_path.name}: requires columns 'Position' and 'UTC'")

    positions = df["Position"].astype(str).tolist()
    times = df["UTC"].astype(str).tolist()

    if len(df) == 0:
        print(f"⚠️  {fr24_csv_path.name}: empty file, skipping")
        return 0

    # sample ~points indices across the track
    idxs = np.linspace(0, len(df) - 1, points, dtype=int)
    sampled_positions = [positions[i] for i in idxs]
    sampled_times = [times[i] for i in idxs]

    rows = []
    for i, (pos, tstamp) in enumerate(zip(sampled_positions, sampled_times), start=1):
        try:
            lat_str, lon_str = [s.strip() for s in pos.split(",")]
            lat, lon = float(lat_str), float(lon_str)
        except Exception:
            # skip bad coordinate rows
            continue

        date_str = _day_str(tstamp)
        hour_str = _iso_floor_to_hour(tstamp)

        # Query a *tight* daily window around the hour to minimize payload
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={date_str}&end_date={date_str}"
            f"&hourly={','.join(PARAMS)}"
            f"&timezone={TIMEZONE}"
        )

        # simple retry
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"❌ Open-Meteo failed for {fr24_csv_path.name} point{i}: {e}")
                    data = {}
                else:
                    time.sleep(1.5)

        hourly = data.get("hourly", {})
        times_list = hourly.get("time", []) or []
        if hour_str in times_list:
            j = times_list.index(hour_str)
            row = {"id": f"point{i}", "lat": lat, "lon": lon, "time": hour_str}
            for p in PARAMS:
                vals = hourly.get(p, [])
                row[p] = (vals[j] if j < len(vals) else None)
            rows.append(row)
        else:
            # Nearest-hour fallback (±1h) if exact hour not present
            # This keeps the pipeline resilient to source quirks.
            nearest = None
            if times_list:
                ser = pd.to_datetime(pd.Series(times_list), utc=True, errors="coerce")
                target = pd.to_datetime(hour_str, utc=True)
                diff = (ser - target).abs()
                k = int(diff.argsort().iloc[0])
                if abs((ser.iloc[k] - target).total_seconds()) <= 3600:  # within 1 hour
                    nearest = k
            if nearest is not None:
                j = nearest
                row = {"id": f"point{i}", "lat": lat, "lon": lon, "time": times_list[j]}
                for p in PARAMS:
                    vals = hourly.get(p, [])
                    row[p] = (vals[j] if j < len(vals) else None)
                rows.append(row)
            # else: skip if no reasonable hour found

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(out_csv_path, index=False)
    return len(out)

def find_predictor_script() -> Path:
    # Prefer the name the user mentioned; fall back to the attached file’s name
    candidates = [
        Path("Weather_history_Predictor.py"),
        Path("weather_history_predictor.py"),
        Path("flight_weather_risk_explained.py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find predictor script. Expected one of: "
        "Weather_history_Predictor.py, weather_history_predictor.py, flight_weather_risk_explained.py"
    )

def run_predictor(predictor_py: Path, in_csv: Path, out_prefix: Path) -> None:
    """
    Call your predictor exactly like you would from CLI:
      python predictor.py --in <in_csv> --out <out_prefix>
    """
    cmd = [sys.executable, str(predictor_py), "--in", str(in_csv), "--out", str(out_prefix)]
    # If you have STD/ATD/STA, append: ["--std", "...Z", "--atd", "...Z", "--sta", "...Z"]
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder of original FR24 CSVs")
    ap.add_argument("--out_dir", default="out", help="Folder for all outputs")
    ap.add_argument("--points", type=int, default=POINTS_DEFAULT, help="Points to sample per flight (default 50)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor_py = find_predictor_script()

    fr24_files = sorted(glob.glob(str(in_dir / "*.csv")))
    if not fr24_files:
        print(f"No CSVs found in {in_dir}")
        sys.exit(0)

    print(f"Found {len(fr24_files)} FR24 file(s). Starting...")
    for path in fr24_files:
        fr24_csv = Path(path)
        stem = fr24_csv.stem  # base name w/o extension

        # 1) Convert → openmeteo flight weather file
        openmeteo_csv = out_dir / f"{stem}_openmeteo_flight_weather_history.csv"
        n = convert_one_fr24_csv(fr24_csv, openmeteo_csv, points=args.points)
        if n == 0:
            print(f"⚠️  Skipping {fr24_csv.name}: produced 0 points")
            continue
        print(f"✅ {fr24_csv.name}: wrote {n} point(s) → {openmeteo_csv.name}")

        # 2) Predict → per_point & summary (predictor will write '<out_prefix>_per_point.csv' and '_summary.csv')
        out_prefix = out_dir / stem
        try:
            run_predictor(predictor_py, openmeteo_csv, out_prefix)
            print(f"✅ Finished: {stem} → {out_prefix}_per_point.csv / {out_prefix}_summary.csv\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Predictor failed for {stem}: {e}\n")

if __name__ == "__main__":
    main()
