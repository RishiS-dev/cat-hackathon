import requests

# Define your Flask API endpoint for the simulator
API_ENDPOINT = "http://127.0.0.1:5001/get_current_data"

# Define thresholds in meters for zones
THRESHOLDS = {
    "near": 3.0,    # less than 3 meters is NEAR
    "medium": 10.0  # between 3 and 10 meters is MEDIUM
    # greater than 10 meters is FAR
}

def classify_zone(distance):
    """Classifies a distance into a zone based on predefined thresholds."""
    if distance < THRESHOLDS["near"]:
        return "NEAR"
    elif distance < THRESHOLDS["medium"]:
        return "MEDIUM"
    else:
        return "FAR"

def get_proximity_zones():
    """
    Fetches sensor data from the simulator API and classifies proximity zones.
    Returns a dictionary with zone info or None if an error occurs.
    """
    try:
        # Fetch data from the Flask API with a 5-second timeout
        response = requests.get(API_ENDPOINT, timeout=5)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        data = response.json()
        
        # Safely get the proximity data
        prox_data = data.get("safety", {}).get("proximity_meters", {})
        if not prox_data:
            print("Error: Proximity data not found in the expected format.")
            return None

        zone_result = {}
        for sensor, value in prox_data.items():
            zone_result[sensor] = {
                "distance_m": value,
                "zone": classify_zone(value)
            }
        return zone_result

    except requests.exceptions.RequestException as e:
        # Handles network errors (e.g., server is down, DNS failure)
        print(f"Error: Could not connect to the API at {API_ENDPOINT}. Please ensure the simulator is running.")
        print(f"Details: {e}")
        return None
    except KeyError as e:
        # Handles cases where the JSON from the API is missing expected keys
        print(f"Error: Unexpected data format from API. Missing key: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    print("Fetching proximity data...")
    zones = get_proximity_zones()

    if zones:
        print("\n--- Proximity Zone Analysis ---")
        for sensor, info in zones.items():
            # Pad sensor name for clean alignment
            print(f"{sensor:<15}: {info['distance_m']:.1f} m  ->  ZONE: {info['zone']}")
        print("-----------------------------\n")

