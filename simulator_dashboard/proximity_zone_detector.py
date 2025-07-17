import requests

# Define your Flask API endpoint
API_ENDPOINT = "http://localhost:5001/get_current_data"

# Define thresholds in meters for zones (adjust as needed)
THRESHOLDS = {
    "near": 3.0,    # less than 3 meters is NEAR
    "medium": 5.0  # between 3 and 10 meters is MEDIUM
    # greater than 10 meters is FAR
}

def classify_zone(distance):
    if distance < THRESHOLDS["near"]:
        return "NEAR"
    elif distance < THRESHOLDS["medium"]:
        return "MEDIUM"
    else:
        return "FAR"

def get_proximity_zones():
    # Fetch data from the Flask API
    response = requests.get(API_ENDPOINT)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")

    data = response.json()
    prox = data["safety"]["proximity_meters"]
    zone_result = {}

    for sensor, value in prox.items():
        zone_result[sensor] = {
            "distance_m": value,
            "zone": classify_zone(value)
        }
    return zone_result

if __name__ == "__main__":
    zones = get_proximity_zones()
    print("Proximity Zones:")
    for sensor, info in zones.items():
        print(f"{sensor}: {info['distance_m']} m -> {info['zone']}")
