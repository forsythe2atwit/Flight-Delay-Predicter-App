from pathlib import Path
import subprocess
import sys

from fr24_csv_converter import convert_fr24_csv


# --------------------------------------------------
# FOLDERS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

FLIGHTS_DIR = ROOT / "flights_csv"

WEATHER_DIR = ROOT / "open_meteo_flight_weather_history"

RISK_DIR = ROOT / "flight_risk_conversions"

PREDICTOR = ROOT / "Weather_history_Predictor.py"


# --------------------------------------------------
# PROCESS ONE FLIGHT
# --------------------------------------------------

def process_flight(flight_csv):

    # Example:
    # DL2644_3ca1e5e6.csv
    #
    # becomes:
    # DL2644

    flight_name = flight_csv.stem.split("_")[0]

    print("\n")
    print("=" * 60)
    print(f"PROCESSING FLIGHT: {flight_name}")
    print("=" * 60)

    # ----------------------------------------------
    # Create output filenames
    # ----------------------------------------------

    weather_file = WEATHER_DIR / f"{flight_name}_weather.csv"

    risk_prefix = RISK_DIR / flight_name

    summary_file = RISK_DIR / f"{flight_name}_summary.csv"

    per_point_file = RISK_DIR / f"{flight_name}_per_point.csv"


    # ----------------------------------------------
    # STEP 1 — FR24 → WEATHER
    # ----------------------------------------------

    print("\n[1/2] Creating weather history...")

    convert_fr24_csv(
        input_csv=flight_csv,
        output_csv=weather_file,
        points=50
    )


    # ----------------------------------------------
    # STEP 2 — WEATHER → RISK
    # ----------------------------------------------

    print("\n[2/2] Calculating flight risk...")

    command = [
        sys.executable,
        str(PREDICTOR),

        "--in",
        str(weather_file),

        "--out",
        str(risk_prefix)
    ]

    subprocess.run(command, check=True)


    # ----------------------------------------------
    # FINISHED
    # ----------------------------------------------

    print(f"\n✓ {flight_name} COMPLETE")

    print(f"Weather:")
    print(f"  {weather_file}")

    print(f"Risk:")
    print(f"  {per_point_file}")
    print(f"  {summary_file}")


# --------------------------------------------------
# PROCESS EVERY FLIGHT
# --------------------------------------------------

def process_all_flights():

    WEATHER_DIR.mkdir(parents=True, exist_ok=True)

    RISK_DIR.mkdir(parents=True, exist_ok=True)

    flights = list(FLIGHTS_DIR.glob("*.csv"))


    if not flights:

        print("No CSV files found in flights_csv/")
        return


    print("\n")
    print("=" * 60)
    print("FLIGHT DELAY BATCH PIPELINE")
    print("=" * 60)

    print(f"\nFound {len(flights)} flight(s).\n")


    successful = 0
    failed = 0


    for number, flight_csv in enumerate(flights, start=1):

        print(
            f"\n[{number}/{len(flights)}] "
            f"{flight_csv.name}"
        )

        try:

            process_flight(flight_csv)

            successful += 1

        except Exception as error:

            failed += 1

            print("\n❌ FLIGHT FAILED")

            print(f"File: {flight_csv.name}")

            print(f"Error: {error}")

            # IMPORTANT:
            # continue instead of crashing the entire batch

            continue


    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)

    print(f"Total flights: {len(flights)}")

    print(f"Successful:    {successful}")

    print(f"Failed:        {failed}")


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":

    process_all_flights()