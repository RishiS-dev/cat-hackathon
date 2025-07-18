from flask import Flask, jsonify, request

app = Flask(__name__)

# In-Memory Data Store with Engine RPM added
sensor_data = {
    "identity": {
        "machine_id": "EXC001",
        "operator_id": "OP1002"
    },
    "status": {
        "ignition_on": True,
        "is_idling": False,
        "engine_hours": 435.2,
        "fuel_percent": 65,
        "engine_temperature_celsius": 95.5,
        "engine_rpm": 800 # <-- ADDED HERE
    },
    "safety": {
        "seatbelt_buckled": True,
        "proximity_meters": {
            "front_left": 15.0, "front_right": 15.0, "side_left_1": 12.5,
            "side_left_2": 12.5, "side_right_1": 12.5, "side_right_2": 12.5,
            "rear_left": 20.0, "rear_right": 20.0
        }
    },
    "environment": {
        "noise_db": 68, "dust_aqi": 45, "air_quality_ppm": 30
    },
    "location": {
        "gps": { "latitude": 11.0173, "longitude": 76.9563 }
    },
    "camera_feeds": {
        "front": "/static/images/view_front.jpg", "rear": "/static/images/view_rear.jpg",
        "left": "/static/images/view_left.jpg", "right": "/static/images/view_right.jpg"
    }
}

@app.route('/get_current_data', methods=['GET'])
def get_data():
    return jsonify(sensor_data)

@app.route('/update_sensor_value', methods=['POST'])
def update_data():
    update_info = request.get_json()
    path = update_info.get('path')
    new_value = update_info.get('value')

    if not isinstance(path, list) or new_value is None:
        return jsonify({"error": "Request must contain a 'path' (list) and 'value'"}), 400

    try:
        current_level = sensor_data
        for key in path[:-1]:
            current_level = current_level[key]
        current_level[path[-1]] = new_value
        return jsonify({"message": f"Successfully updated path {path}", "new_state": sensor_data}), 200
    except (KeyError, TypeError):
        return jsonify({"error": f"Invalid path: {path}"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
