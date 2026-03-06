import os
import logging
from flask import Flask, render_template, request
import requests

from fr24sdk.client import Client
from fr24sdk.models.flight import FlightSummaryLight

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
FR24_API_TOKEN = os.environ.get("FR24_API_TOKEN", "0199798a-b3f4-71c2-bfcd-adbe3d03764f|ZgPsVTQQmt02NgYLWwB4mtorNumAzUi5oKomJCnH6433422b")

client = Client(api_token=FR24_API_TOKEN)


result = client.live.flight_positions.get_light(bounds="50.682,46.218, 14.422,22.243") # N, S, W, E
print(result)


from math import radians, cos, sin, asin, sqrt, atan2
from typing import Iterable, Dict, Any

import time

POLLING_SECONDS = 5
def monitor_geofence() -> None:
    while True:
        try:
            flights = client.live.flight_positions.get_light(bounds=BOUNDS)
            inside = list(flights_in_circle(flights.data))
            if inside:
                process_alerts(inside)
            time.sleep(POLLING_SECONDS)
        except Exception as exc:
            logging.error("Error during polling: %s", exc)
            time.sleep(POLLING_SECONDS)

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0 # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
def is_inside_circle (flat: float, flon: float) -> bool:
    return haversine_km(flat, flon, CENTER_LAT, CENTER_LON) <= RADIUS

def flights_in_circle(flights: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for flight in flights:
        lat = flight.lat
        lon = flight.lon
        if lat is not None and lon is not None and is_inside_circle(lat, lon):
            yield flight

alerted: set[str] = set()

def send_alert(flight: dict) -> None:
    logging.info("ALERT  ✈️  Flight %s entered the area", flight.fr24_id)

def process_alerts(flights: FlightSummaryLight) -> None:
    for flight in flights:
        flight_id = flight.fr24_id
        if not flight_id:
            continue
        if flight_id in alerted:
            continue
        send_alert(flight)
        alerted.add(flight_id)


app = Flask(__name__)

@app.route("/")
def index():
    flights = client.live.flight_positions.get_light(bounds=BOUNDS)
    return render_template("index.html", flights=flights.data)


FR24_API_TOKEN = "0199798a-b3f4-71c2-bfcd-adbe3d03764f|ZgPsVTQQmt02NgYLWwB4mtorNumAzUi5oKomJCnH6433422b"

CENTER_LAT = 42.3555 # Example: Boston latitude
CENTER_LON = -71.0565
RADIUS = 50 # circle radius

def make_bounds(center_lat: float, center_lon: float, lat_delta: float = 1.0, lon_delta: float = 1.0) -> str:
    north = center_lat + lat_delta
    south = center_lat - lat_delta
    west = center_lon - lon_delta
    east = center_lon + lon_delta
    return f"{north},{south},{west},{east}"
BOUNDS = make_bounds(CENTER_LAT, CENTER_LON)
flights = client.live.flight_positions.get_light(bounds=BOUNDS)

if __name__ == '__main__':
    app.run(debug=True)