from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fr24_csv_converter import convert_fr24_csv


ROOT = Path(__file__).resolve().parent
FLIGHTS_DIR = ROOT / "flights_csv"
WEATHER_DIR = ROOT / "open_meteo_flight_weather_history"
RISK_DIR = ROOT / "flight_risk_conversions"
PREDICTOR = ROOT / "Weather_history_Predictor.py"


def newest_fr24_csv(folder: Path) -> Path:
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(
            f"Created {folder}. Put an FR24 CSV inside it, then run this command again."
        )

    candidates = [
        p for p in folder.glob("*.csv")
        if p.is_file() and not p.name.startswith(".")
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found in {folder}. Download an FR24 CSV into that folder first."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def flight_name(csv_path: Path) -> str:
    # FR24 names commonly look like CX811_3e70d9c1.csv.
    # Preserve the callsign/flight-number portion and remove the FR24 hex-like suffix.
    stem = csv_path.stem
    cleaned = re.sub(r"_[0-9a-fA-F]{6,}$", "", stem)
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", cleaned).strip("_")
    return cleaned or stem


def validate_fr24_csv(path: Path) -> None:
    preview = pd.read_csv(path, nrows=5)
    missing = {"Position", "UTC"} - set(preview.columns)
    if missing:
        raise ValueError(
            f"{path.name} does not look like the expected FR24 track CSV. "
            f"Missing: {', '.join(sorted(missing))}"
        )


def run_predictor(weather_csv: Path, risk_prefix: Path) -> None:
    if not PREDICTOR.exists():
        raise FileNotFoundError(f"Missing predictor script: {PREDICTOR}")

    command = [
        sys.executable,
        str(PREDICTOR),
        "--in",
        str(weather_csv),
        "--out",
        str(risk_prefix),
    ]

    print("\nCalculating per-point risk and WDI...")
    subprocess.run(command, check=True)


def show_summary(summary_csv: Path) -> None:
    if not summary_csv.exists():
        print(f"Analysis finished, but summary file was not found: {summary_csv}")
        return

    summary = pd.read_csv(summary_csv).iloc[0]

    print("\n" + "=" * 52)
    print("FLIGHT WEATHER RISK RESULT")
    print("=" * 52)
    print(f"Departure risk : {summary.get('dep_score_mean', 'N/A')}")
    print(f"Enroute risk   : {summary.get('enroute_score_mean', 'N/A')}")
    print(f"Arrival risk   : {summary.get('arr_score_mean', 'N/A')}")
    print(f"WDI            : {summary.get('WDI_0_100', 'N/A')} / 100")
    print(f"Heuristic delay: {summary.get('heuristic_delay_min', 'N/A')} min")
    print("=" * 52)


def analyze(csv_path: Path, points: int = 50) -> None:
    csv_path = csv_path.resolve()
    validate_fr24_csv(csv_path)

    name = flight_name(csv_path)

    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    RISK_DIR.mkdir(parents=True, exist_ok=True)

    weather_csv = WEATHER_DIR / f"{name}_weather.csv"
    risk_prefix = RISK_DIR / name
    per_point_csv = RISK_DIR / f"{name}_per_point.csv"
    summary_csv = RISK_DIR / f"{name}_summary.csv"

    print(f"\nDetected flight CSV: {csv_path.name}")
    print(f"Flight name:         {name}")
    print("\n[1/2] Building trajectory weather history...")
    convert_fr24_csv(csv_path, weather_csv, points=points)

    print("\n[2/2] Converting weather into flight risk...")
    run_predictor(weather_csv, risk_prefix)

    show_summary(summary_csv)

    print("\nOutputs:")
    print(f"  Weather   : {weather_csv}")
    print(f"  Per-point : {per_point_csv}")
    print(f"  Summary   : {summary_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatically analyze the newest FR24 CSV and create flight-risk outputs."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        help="Optional FR24 CSV path. If omitted, newest CSV in flights_csv/ is used.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=50,
        help="Trajectory weather sample count (default: 50).",
    )
    args = parser.parse_args()

    try:
        selected = Path(args.csv) if args.csv else newest_fr24_csv(FLIGHTS_DIR)
        if not selected.is_absolute():
            selected = (ROOT / selected).resolve()
        analyze(selected, points=args.points)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
