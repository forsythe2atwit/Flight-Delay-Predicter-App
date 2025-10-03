import os
import pandas as pd
from fr24sdk.client import Client

FR24_API_TOKEN = os.getenv("FR24_API_TOKEN", "01999c69-7045-7139-bf98-3f59f53c39be|VSVhs828XYGnu8zRJuKoiPlowekLIei3kFP6UXwe5e854392")
client = Client(api_token=FR24_API_TOKEN)

OUTPUT_DIR = "fr24_flights"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_flight(flight_id):
    """Download a single flight trace and save as CSV."""
    trace = client.flights.get_history_by_id(flight_id)
    if not trace or "data" not in trace:
        print(f"⚠️ No data for {flight_id}")
        return None
    
    df = pd.DataFrame(trace["data"])
    if df.empty:
        return None

    # Save as CSV
    out_file = os.path.join(OUTPUT_DIR, f"{flight_id}.csv")
    df.to_csv(out_file, index=False)
    print(f"✅ Saved {out_file}")
    return out_file

def download_many(flight_ids):
    """Download many flights in bulk."""
    for fid in flight_ids:
        try:
            download_flight(fid)
        except Exception as e:
            print(f"❌ Failed {fid}: {e}")

if __name__ == "__main__":
    # Example list of FR24 flight IDs (replace with real IDs)
    flights = ["DL527",
    "DL1130",
    "DL1443",
    "DL1470",
    "DL1660",
    "DL2471",
    "DL2908",
    "DL2911",
    "DL3023",
    "DL8804",
    "DL2017",
    "DL2024",
    "DL5827",
    "DL1101",
    "DL1826",
    "DL1843",
    "DL1857",
    "DL8788",
    "DL1517",
    "DL2016",
    "DL1229",
    "DL1263",
    "DL1657",
    "DL3847",
    "DL4763",
    "DL5027",
    "DL5133",
    "DL5408",
    "DL563",
    "DL572",
    "DL2068",
    "DL5230",
    "DL2882",
    "DL2903",
    "DL2993",
    "DL2256"]
    download_many(flights)
