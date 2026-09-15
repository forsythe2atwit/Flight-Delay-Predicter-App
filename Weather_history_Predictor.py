#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weather_history_Predictor.py
---------------------------
Reads an Open-Meteo flight weather-history CSV and produces:
  - <out_prefix>_per_point.csv
  - <out_prefix>_summary.csv

Usage:
  python Weather_history_Predictor.py --in open_meteo_flight_weather_history/DAL603_openmeteo.csv --out out/DAL603

This is your WDI + heuristic delay pipeline (interpretable).
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = (0.35, 0.25, 0.20, 0.15, 0.05)

WDI_BREAKS = np.array([0, 25, 50, 75, 100], dtype=float)
MINUTES_AT_BREAKS = np.array([0, 5, 15, 30, 60], dtype=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Weather Delay Index from weather CSV.")
    p.add_argument("--in", dest="in_csv", required=True, help="Input Open-Meteo flight weather-history CSV")
    p.add_argument("--out", dest="out_prefix", required=True, help="Output prefix (no extension)")

    # Optional schedule anchors
    p.add_argument("--std", dest="std_iso", default=None, help="Scheduled Time of Departure (UTC ISO)")
    p.add_argument("--atd", dest="atd_iso", default=None, help="Actual Time of Departure (UTC ISO)")
    p.add_argument("--sta", dest="sta_iso", default=None, help="Scheduled Time of Arrival (UTC ISO)")

    # Tunables
    p.add_argument("--weights", nargs=5, type=float, default=list(DEFAULT_WEIGHTS),
                   metavar=("W_PRECIP", "W_VIS", "W_WIND", "W_FOG", "W_SNOW"))
    p.add_argument("--dep_window_min", type=int, default=60)
    p.add_argument("--arr_window_min", type=int, default=60)
    p.add_argument("--phase_pct", type=float, default=0.20)
    return p.parse_args()


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan


def compute_per_point_risks(df: pd.DataFrame,
                            weights: Tuple[float, float, float, float, float]) -> pd.DataFrame:
    # VISIBILITY risk: <10km rises to 100
    vis_m = pd.to_numeric(df.get("visibility", np.nan), errors="coerce")
    vis_risk = (1.0 - (vis_m / 10000.0)).clip(0.0, 1.0) * 100.0

    # PRECIP risk: 0.7*prob + 0.3*intensity
    prob = pd.to_numeric(df.get("precipitation_probability", 0.0), errors="coerce").fillna(0.0) / 100.0
    rain = pd.to_numeric(df.get("rain", 0.0), errors="coerce").fillna(0.0)
    showers = pd.to_numeric(df.get("showers", 0.0), errors="coerce").fillna(0.0)
    snowfall = pd.to_numeric(df.get("snowfall", 0.0), errors="coerce").fillna(0.0)
    intensity_norm = (rain + showers + snowfall).clip(0.0, 8.0) / 8.0
    precip_risk = (0.7 * prob + 0.3 * intensity_norm) * 100.0

    # WIND risk: gusts preferred, cap at 100 km/h
    gusts = pd.to_numeric(df.get("windgusts_10m", np.nan), errors="coerce")
    windspeed = pd.to_numeric(df.get("windspeed_10m", 0.0), errors="coerce")
    gusts = gusts.fillna(windspeed)
    wind_risk = (gusts / 100.0).clip(0.0, 1.0) * 100.0

    # FOG proxy: spread <=2 => 100, spread >=6 => 0
    T = pd.to_numeric(df.get("apparent_temperature", np.nan), errors="coerce")
    Td = pd.to_numeric(df.get("dew_point_2m", np.nan), errors="coerce")
    spread = T - Td
    fog_risk = (1.0 - ((spread - 2.0) / 4.0)).clip(0.0, 1.0) * 100.0

    # SNOW risk: flag + intensity
    snow_depth = pd.to_numeric(df.get("snow_depth", 0.0), errors="coerce").fillna(0.0)
    snow_flag = ((snowfall > 0.0) | (snow_depth > 0.0)).astype(int)
    snow_risk = snow_flag * 60.0 + (snowfall.clip(0.0, 5.0) / 5.0) * 40.0

    w_precip, w_vis, w_wind, w_fog, w_snow = weights
    severity = (
        w_precip * precip_risk +
        w_vis    * vis_risk +
        w_wind   * wind_risk +
        w_fog    * fog_risk +
        w_snow   * snow_risk
    )

    return pd.DataFrame({
        "precip_risk": precip_risk.round(2),
        "vis_risk": vis_risk.round(2),
        "wind_risk": wind_risk.round(2),
        "fog_risk": fog_risk.round(2),
        "snow_risk": snow_risk.round(2),
        "severity_0_100": severity.round(2),
        "precip_intensity_norm_0_1": intensity_norm.round(3),
        "precip_probability_0_1": prob.round(3),
    })


def choose_phase_windows(df_time_sorted: pd.DataFrame,
                         std_iso: str | None,
                         atd_iso: str | None,
                         sta_iso: str | None,
                         dep_window_min: int = 60,
                         arr_window_min: int = 60,
                         phase_pct: float = 0.20):
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
    if np.isnan(wdi):
        return np.nan
    return float(np.interp(wdi, WDI_BREAKS, MINUTES_AT_BREAKS))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.in_csv)

    ensure_cols(df, ["time", "lat", "lon"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)

    weather_cols = [
        "visibility", "precipitation_probability",
        "rain", "showers", "snowfall", "snow_depth",
        "windspeed_10m", "windgusts_10m",
        "apparent_temperature", "dew_point_2m",
    ]
    ensure_cols(df, weather_cols)

    weights = tuple(args.weights)
    risks = compute_per_point_risks(df, weights)
    per_point = pd.concat([df[["time", "lat", "lon"]].reset_index(drop=True), risks], axis=1)

    dep_mask, enroute_mask, arr_mask, used_schedule = choose_phase_windows(
        df_time_sorted=per_point,
        std_iso=args.std_iso,
        atd_iso=args.atd_iso,
        sta_iso=args.sta_iso,
        dep_window_min=args.dep_window_min,
        arr_window_min=args.arr_window_min,
        phase_pct=args.phase_pct,
    )

    dep_score = per_point.loc[dep_mask, "severity_0_100"].mean()
    enr_score = per_point.loc[enroute_mask, "severity_0_100"].mean()
    arr_score = per_point.loc[arr_mask, "severity_0_100"].mean()

    W_DEP, W_ENR, W_ARR = 0.4, 0.2, 0.4
    WDI = float(np.nansum([W_DEP * dep_score, W_ENR * enr_score, W_ARR * arr_score]))
    delay_min = wdi_to_delay_minutes(WDI)

    per_point_path = f"{args.out_prefix}_per_point.csv"
    summary_path = f"{args.out_prefix}_summary.csv"
    per_point.to_csv(per_point_path, index=False)

    summary = pd.DataFrame([{
        "n_points": int(len(per_point)),
        "dep_points": int(dep_mask.sum()),
        "enroute_points": int(enroute_mask.sum()),
        "arr_points": int(arr_mask.sum()),
        "used_schedule_windows": bool(used_schedule),
        "dep_score_mean": None if pd.isna(dep_score) else round(float(dep_score), 2),
        "enroute_score_mean": None if pd.isna(enr_score) else round(float(enr_score), 2),
        "arr_score_mean": None if pd.isna(arr_score) else round(float(arr_score), 2),
        "WDI_0_100": None if pd.isna(WDI) else round(float(WDI), 2),
        "heuristic_delay_min": None if pd.isna(delay_min) else round(float(delay_min), 1),
        "weights_precip_vis_wind_fog_snow": weights,
    }])
    summary.to_csv(summary_path, index=False)

    print(f"✓ Saved per-point risks: {per_point_path}")
    print(f"✓ Saved flight summary:  {summary_path}")
    if not np.isnan(WDI):
        print(f"WDI: {WDI:.2f}  |  heuristic delay: ~{delay_min:.1f} min")


if __name__ == "__main__":
    main()