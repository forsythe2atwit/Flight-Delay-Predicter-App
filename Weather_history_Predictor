#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flight_weather_risk_explained.py
--------------------------------
GOAL
====
Given a CSV of weather conditions sampled along a flight's path (e.g., 50 points),
compute:
  1) Per-point risk components (0..100), and a blended "severity" (0..100)
  2) A flight-level Weather Delay Index (WDI, 0..100) that weights departure and arrival
  3) A simple, interpretable mapping from WDI -> expected delay minutes (heuristic)

WHY THIS MATTERS
================
- Flights are most delay-sensitive near departure and arrival (pushback, taxi, takeoff,
  approach, landing, gate). The pipeline assigns heavier weights to those phases.
- We keep the math simple, transparent, and tunable. You can later calibrate with real
  labels (actual delays) to refine thresholds/weights.

INPUT CSV (columns expected; extra columns are fine)
====================================================
Minimal required columns:
  - time (ISO 8601 string in UTC, e.g., 2025-09-27T01:00:00Z)
  - lat, lon (floats; not strictly used in scoring, but helpful for debugging/plotting)

Weather columns (Open-Meteo style; we'll coerce/assume units):
  - visibility                    (meters; 10,000 m = "good", lower is worse)
  - precipitation_probability     (% 0..100)
  - rain, showers, snowfall       (mm/h)
  - snow_depth                    (cm or m depending on source; we only use >0 as a flag)
  - windspeed_10m, windgusts_10m  (km/h; gusts preferred)
  - apparent_temperature          (°C; proxy for surface T)
  - dew_point_2m                  (°C; surface dewpoint)

OPTIONAL: schedule anchors (if provided as CLI flags)
=====================================================
- STD: Scheduled Time of Departure (UTC ISO)
- ATD: Actual Time of Departure (UTC ISO)
- STA: Scheduled Time of Arrival   (UTC ISO)

If these are provided, the script will:
  - Make the "departure" window: [STD - dep_window_min, ATD] (or STD + dep_window_min)
  - Make the "arrival" window:   [STA - arr_window_min, STA + arr_window_min]

If not provided, we approximate:
  - First ~phase_pct (default 20%) of points = departure phase
  - Last ~phase_pct (default 20%) of points  = arrival phase

OUTPUTS
=======
- <out_prefix>_per_point.csv : time, lat, lon + risk components + severity (0..100)
- <out_prefix>_summary.csv   : 1 row with WDI, per-phase means, and heuristic delay

USAGE EXAMPLES
==============
python flight_weather_risk_explained.py --in openmeteo_flight_weather_history.csv --out flight1

# With schedule anchors (recommended if you have them):
python flight_weather_risk_explained.py \
  --in openmeteo_flight_weather_history.csv \
  --out flight1 \
  --std 2025-09-27T18:55:00Z \
  --atd 2025-09-27T23:01:00Z \
  --sta 2025-09-28T00:50:00Z

# Change risk weights (precip, vis, wind, fog, snow must sum ≈ 1):
python flight_weather_risk_explained.py --in openmeteo_flight_weather_history.csv --out flight1 \
  --weights 0.30 0.25 0.25 0.10 0.10

# Change window sizes:
python flight_weather_risk_explained.py --in openmeteo_flight_weather_history.csv --out flight1 \
  --dep_window_min 90 --arr_window_min 90
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------
# 1) GLOBAL CONFIG — default, interpretable settings (tunable later)
# --------------------------------------------------------------------

# Component weights for blending per-point risks into a single severity score.
# Order: (precip, visibility, wind, fog, snow). These should sum ~ 1.0
DEFAULT_WEIGHTS = (0.35, 0.25, 0.20, 0.15, 0.05)

# Simple, interpretable mapping from the final WDI (0..100) to *expected* delay minutes.
# Think of this as a communication device for ops teams; calibrate later with real labels.
WDI_BREAKS = np.array([0, 25, 50, 75, 100], dtype=float)
MINUTES_AT_BREAKS = np.array([0, 5, 15, 30, 60], dtype=float)


# --------------------------------------------------------------------
# 2) CLI ARGUMENTS
# --------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Weather Delay Index from weather CSV.")
    p.add_argument("--in", dest="in_csv", required=True,
                   help="Input CSV with columns: time, lat, lon, visibility, precipitation_probability, "
                        "rain, showers, snowfall, snow_depth, windspeed_10m, windgusts_10m, "
                        "apparent_temperature, dew_point_2m")
    p.add_argument("--out", dest="out_prefix", required=True,
                   help="Output prefix (no extension); creates <out>_per_point.csv and <out>_summary.csv")

    # Optional schedule anchors (UTC ISO strings). If omitted, index uses first/last 20% of points.
    p.add_argument("--std", dest="std_iso", default=None, help="Scheduled Time of Departure (UTC ISO)")
    p.add_argument("--atd", dest="atd_iso", default=None, help="Actual Time of Departure (UTC ISO)")
    p.add_argument("--sta", dest="sta_iso", default=None, help="Scheduled Time of Arrival (UTC ISO)")

    # Optional tuning of risk weights and phase windows
    p.add_argument("--weights", nargs=5, type=float, default=list(DEFAULT_WEIGHTS),
                   metavar=("W_PRECIP", "W_VIS", "W_WIND", "W_FOG", "W_SNOW"),
                   help="Per-point risk blend weights; should sum ≈ 1.0 (default 0.35 0.25 0.20 0.15 0.05)")
    p.add_argument("--dep_window_min", type=int, default=60,
                   help="Minutes before STD for departure window (if STD/ATD provided). Default 60.")
    p.add_argument("--arr_window_min", type=int, default=60,
                   help="Minutes around STA for arrival window (before/after). Default 60.")
    p.add_argument("--phase_pct", type=float, default=0.20,
                   help="If no schedule anchors, the fraction of points used for dep/arr windows. Default 0.20")

    return p.parse_args()


# --------------------------------------------------------------------
# 3) HELPER FUNCTIONS
# --------------------------------------------------------------------
def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    """Ensure required columns exist (create as NaN if missing)."""
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan


def compute_per_point_risks(df: pd.DataFrame,
                            weights: Tuple[float, float, float, float, float]) -> pd.DataFrame:
    """
    Convert raw weather variables into 5 component risks (0..100) + a blended "severity" (0..100).

    IMPORTANT UNIT ASSUMPTIONS (tune if your provider differs):
    - visibility: meters (10,000 m is considered "good", 0..100 risk increases as vis drops)
    - precipitation_probability: % (0..100)
    - rain, showers, snowfall: mm/h (we cap sum at 8 mm/h for normalization)
    - windspeed_10m, windgusts_10m: km/h (we cap risk at 100 km/h gusts)
    - apparent_temperature, dew_point_2m: °C
    - snow_depth: only used as a boolean (>0 means "snow present")

    RATIONALE for each component:
    - precip_risk balances "will it occur?" (probability) vs "how heavy?" (intensity)
    - vis_risk penalizes low visibility linearly below 10 km (think LVP)
    - wind_risk uses gusts as primary driver; strong gusts disrupt ops
    - fog_risk uses small T - Td spread as a proxy for fog/low ceiling
    - snow_risk flags snow/ice as a special ops constraint (de-icing, runway condition)
    """

    # --- VISIBILITY RISK (0..100) ---
    # 10,000 m (10 km) and better -> low risk; below that -> increasing risk linearly
    vis_m = pd.to_numeric(df["visibility"], errors="coerce")
    vis_risk = (1.0 - (vis_m / 10000.0)).clip(lower=0.0, upper=1.0) * 100.0

    # --- PRECIPITATION RISK (0..100) ---
    # Two signals:
    #   (A) precip probability (0..100%) → scaled to [0,1]
    #   (B) intensity proxy = (rain + showers + snowfall) mm/h, capped at 8 mm/h, normalized to [0,1]
    # Heuristic blend: 0.7 * probability + 0.3 * intensity
    prob = pd.to_numeric(df.get("precipitation_probability", 0.0), errors="coerce").fillna(0.0) / 100.0
    rain = pd.to_numeric(df.get("rain", 0.0), errors="coerce").fillna(0.0)
    showers = pd.to_numeric(df.get("showers", 0.0), errors="coerce").fillna(0.0)
    snowfall = pd.to_numeric(df.get("snowfall", 0.0), errors="coerce").fillna(0.0)
    intensity_norm = (rain + showers + snowfall).clip(0.0, 8.0) / 8.0  # 8 mm/h = "very heavy" normalization
    precip_risk = (0.7 * prob + 0.3 * intensity_norm) * 100.0

    # --- WIND RISK (0..100) ---
    # Use gusts when available; cap at 100 km/h to define an intuitive 0..100 scale
    gusts = pd.to_numeric(df.get("windgusts_10m", np.nan), errors="coerce")
    windspeed = pd.to_numeric(df.get("windspeed_10m", 0.0), errors="coerce")
    gusts = gusts.fillna(windspeed)  # fall back to sustained if gusts missing
    wind_risk = (gusts / 100.0).clip(0.0, 1.0) * 100.0

    # --- FOG / LOW CEILING PROXY (0..100) ---
    # Small temperature-dewpoint spread indicates saturation → fog/low ceiling risk.
    # Map: spread <= 2°C → 100 risk; spread >= 6°C → 0 risk; linear in between.
    T = pd.to_numeric(df.get("apparent_temperature", np.nan), errors="coerce")
    Td = pd.to_numeric(df.get("dew_point_2m", np.nan), errors="coerce")
    spread = T - Td
    fog_risk = (1.0 - ((spread - 2.0) / 4.0)).clip(0.0, 1.0) * 100.0

    # --- SNOW / ICE RISK (0..100) ---
    # If any snowfall or snow depth present, assign a strong baseline (60), then add intensity up to +40.
    snow_depth = pd.to_numeric(df.get("snow_depth", 0.0), errors="coerce").fillna(0.0)
    snow_flag = ((snowfall > 0.0) | (snow_depth > 0.0)).astype(int)
    snow_risk = snow_flag * 60.0 + (snowfall.clip(0.0, 5.0) / 5.0) * 40.0  # 5 mm/h snowfall ~ adds +40 (capped)

    # --- BLEND COMPONENTS INTO "SEVERITY" ---
    # Weighted sum; weights are chosen heuristically and can be tuned by learning from data.
    w_precip, w_vis, w_wind, w_fog, w_snow = weights
    severity = (
        w_precip * precip_risk +
        w_vis    * vis_risk +
        w_wind   * wind_risk +
        w_fog    * fog_risk +
        w_snow   * snow_risk
    )

    # Pack results into a new DataFrame (rounded for readability)
    out = pd.DataFrame({
        "precip_risk": precip_risk.round(2),
        "vis_risk":    vis_risk.round(2),
        "wind_risk":   wind_risk.round(2),
        "fog_risk":    fog_risk.round(2),
        "snow_risk":   snow_risk.round(2),
        "severity_0_100": severity.round(2),
        # (Optional) export intensity_norm if you want to audit precip calculation:
        "precip_intensity_norm_0_1": intensity_norm.round(3),
        "precip_probability_0_1": prob.round(3),
    })
    return out


def choose_phase_windows(df_time_sorted: pd.DataFrame,
                         std_iso: str | None,
                         atd_iso: str | None,
                         sta_iso: str | None,
                         dep_window_min: int = 60,
                         arr_window_min: int = 60,
                         phase_pct: float = 0.20) -> Tuple[pd.Series, pd.Series, pd.Series, bool]:
    """
    Build boolean masks for DEPARTURE / EN-ROUTE / ARRIVAL segments.

    Two modes:
      A) If schedule anchors provided:
           dep window: [STD - dep_window_min, ATD] (or STD + dep_window_min if ATD missing)
           arr window: [STA - arr_window_min, STA + arr_window_min]
         Everything else = en-route.

      B) If no anchors:
           First phase_pct of points = departure
           Last  phase_pct of points = arrival
           Middle = en-route

    Returns:
      dep_mask, enroute_mask, arr_mask, used_schedule (bool)
    """
    t = pd.to_datetime(df_time_sorted["time"], utc=True, errors="coerce")
    n = len(t)

    used_schedule = bool(std_iso) and bool(sta_iso)
    if used_schedule:
        STD = pd.to_datetime(std_iso, utc=True, errors="coerce")
        ATD = pd.to_datetime(atd_iso, utc=True, errors="coerce") if atd_iso else None
        STA = pd.to_datetime(sta_iso, utc=True, errors="coerce")

        dep_start = STD - pd.Timedelta(minutes=dep_window_min)
        dep_end = ATD if ATD is not None else (STD + pd.Timedelta(minutes=dep_window_min))
        arr_start = STA - pd.Timedelta(minutes=arr_window_min)
        arr_end = STA + pd.Timedelta(minutes=arr_window_min)

        dep_mask = (t >= dep_start) & (t <= dep_end)
        arr_mask = (t >= arr_start) & (t <= arr_end)
    else:
        k = max(1, int(phase_pct * n))
        dep_mask = pd.Series(False, index=t.index)
        dep_mask.iloc[:k] = True
        arr_mask = pd.Series(False, index=t.index)
        arr_mask.iloc[-k:] = True

    enroute_mask = ~(dep_mask | arr_mask)
    return dep_mask, enroute_mask, arr_mask, used_schedule


def wdi_to_delay_minutes(wdi: float) -> float:
    """
    Interpolate WDI (0..100) to a simple expected delay minutes.
    This is NOT a predictive model — it's an interpretable yardstick.
    Calibrate MINUTES_AT_BREAKS with your real labels for better fidelity.
    """
    if np.isnan(wdi):
        return np.nan
    return float(np.interp(wdi, WDI_BREAKS, MINUTES_AT_BREAKS))


# --------------------------------------------------------------------
# 4) MAIN: glue everything together
# --------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    in_csv = args.in_csv
    out_prefix = args.out_prefix
    weights = tuple(args.weights)

    # ---------- Load & normalize ----------
    df = pd.read_csv(in_csv)
    # Ensure essential columns exist (create as NaN if missing)
    ensure_cols(df, ["time", "lat", "lon"])

    # Normalize time to UTC pandas datetime; sort by time (critical for phase masks)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)

    # Ensure weather columns exist (we'll coerce to numeric in compute_per_point_risks)
    weather_cols = [
        "visibility", "precipitation_probability",
        "rain", "showers", "snowfall", "snow_depth",
        "windspeed_10m", "windgusts_10m",
        "apparent_temperature", "dew_point_2m",
    ]
    ensure_cols(df, weather_cols)

    # ---------- Compute per-point risks ----------
    risks = compute_per_point_risks(df, weights)
    per_point = pd.concat([df[["time", "lat", "lon"]].reset_index(drop=True), risks], axis=1)

    # ---------- Build phase windows ----------
    dep_mask, enroute_mask, arr_mask, used_schedule = choose_phase_windows(
        df_time_sorted=per_point,
        std_iso=args.std_iso,
        atd_iso=args.atd_iso,
        sta_iso=args.sta_iso,
        dep_window_min=args.dep_window_min,
        arr_window_min=args.arr_window_min,
        phase_pct=args.phase_pct
    )

    # ---------- Aggregate to flight-level ----------
    dep_score = per_point.loc[dep_mask, "severity_0_100"].mean()
    enr_score = per_point.loc[enroute_mask, "severity_0_100"].mean()
    arr_score = per_point.loc[arr_mask, "severity_0_100"].mean()

    # Phase weights (interpretable, not learned): dep=0.4, enroute=0.2, arrival=0.4
    W_DEP, W_ENR, W_ARR = 0.4, 0.2, 0.4
    WDI = float(np.nansum([W_DEP * dep_score, W_ENR * enr_score, W_ARR * arr_score]))

    # Heuristic delay minutes for communication (calibrate later)
    delay_min = wdi_to_delay_minutes(WDI)

    # ---------- Save outputs ----------
    per_point_path = f"{out_prefix}_per_point.csv"
    summary_path = f"{out_prefix}_summary.csv"
    per_point.to_csv(per_point_path, index=False)

    summary = pd.DataFrame([{
        # bookkeeping
        "n_points": int(len(per_point)),
        "dep_points": int(dep_mask.sum()),
        "enroute_points": int(enroute_mask.sum()),
        "arr_points": int(arr_mask.sum()),
        "used_schedule_windows": used_schedule,

        # phase scores (mean severities)
        "dep_score_mean": None if pd.isna(dep_score) else round(float(dep_score), 2),
        "enroute_score_mean": None if pd.isna(enr_score) else round(float(enr_score), 2),
        "arr_score_mean": None if pd.isna(arr_score) else round(float(arr_score), 2),

        # overall index
        "WDI_0_100": None if pd.isna(WDI) else round(float(WDI), 2),

        # interpretable mapping (not a prediction; calibrate later)
        "heuristic_delay_min": None if pd.isna(delay_min) else round(float(delay_min), 1),

        # record the weights used
        "weights_precip_vis_wind_fog_snow": weights
    }])
    summary.to_csv(summary_path, index=False)

    # ---------- Console summary ----------
    print(f"✓ Saved per-point risks: {per_point_path}")
    print(f"✓ Saved flight summary:  {summary_path}")
    if not np.isnan(WDI):
        print(f"Weather Delay Index (WDI): {WDI:.2f}  |  Heuristic delay: ~{delay_min:.1f} min")
        if used_schedule:
            print("(Schedule-anchored windows were used.)")
        else:
            print("(Approximate windows used: first/last {:.0f}% of points.)".format(args.phase_pct * 100))
    else:
        print("WDI could not be computed (missing or invalid data).")


if __name__ == "__main__":
    main()
