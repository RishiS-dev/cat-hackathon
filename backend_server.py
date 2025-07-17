import os
import psycopg2
import psycopg2.extras # Needed for JSONB
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS # Make sure this is installed: pip install Flask-Cors

# --- Initialization ---
load_dotenv()
app = Flask(__name__)

# --- ROBUST CORS SETUP ---
# This is a more explicit and reliable way to handle CORS.
# It tells the server to allow all origins ('*') to access all API routes.
CORS(app, resources={r"/api/*": {"origins": "*"}})

# In-memory store for the demo's active session state
active_shift = {
    "shift_id": None,
    "task_id": None,
    "geofence_wkt": None # Well-Known Text format for the polygon
}

# --- Database Configuration ---
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"), "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# --- Alert Thresholds ---
ALERT_THRESHOLDS = {
    "PROXIMITY_NEAR": 3.0,
    "HIGH_NOISE": 90.0,
    "HIGH_AQI": 200.0,
    "HIGH_ENGINE_TEMP": 115.0
}

# --- API Endpoints ---
@app.route('/')
def index():
    # A simple test route to confirm the server is running
    return "Backend server is running!"

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO work_shifts (operator_id, machine_id) VALUES (%s, %s) RETURNING shift_id;",
        (data.get('operator_id'), data.get('machine_id'))
    )
    shift_id = cur.fetchone()[0]
    conn.commit()
    active_shift["shift_id"] = shift_id
    cur.close()
    conn.close()
    return jsonify({"message": "Login successful", "shift_id": shift_id}), 200

@app.route('/api/schedule', methods=['POST'])
def post_schedule():
    tasks = request.get_json()
    conn = get_db_connection()
    cur = conn.cursor()
    task_ids = []
    for task in tasks:
        points_str = ", ".join([f"{p[0]} {p[1]}" for p in task['geofence_points']])
        polygon_wkt = f"POLYGON(({points_str}))"
        cur.execute(
            """
            INSERT INTO scheduled_tasks (assigned_date, operator_id, machine_id, task_type, load_cycles_planned, geofence, task_inputs)
            VALUES (%s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326), %s) RETURNING task_id;
            """,
            (
                task['assigned_date'], task['operator_id'], task['machine_id'],
                task['task_type'], task['load_cycles_planned'],
                polygon_wkt, psycopg2.extras.Json(task['task_inputs'])
            )
        )
        task_ids.append(cur.fetchone()[0])
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"message": f"{len(tasks)} tasks scheduled.", "task_ids": task_ids}), 201

@app.route('/api/set_task', methods=['POST'])
def set_task():
    data = request.get_json()
    task_id = data.get('task_id')
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT ST_AsText(geofence) FROM scheduled_tasks WHERE task_id = %s;", (task_id,))
    result = cur.fetchone()
    if result:
        active_shift['task_id'] = task_id
        active_shift['geofence_wkt'] = result[0]
        cur.execute("UPDATE work_shifts SET active_task_id = %s WHERE shift_id = %s;", (task_id, active_shift['shift_id']))
        conn.commit()
    cur.close()
    conn.close()
    return jsonify({"message": f"Active task set to {task_id}"})

@app.route('/api/live_status', methods=['GET'])
def get_live_status():
    if not active_shift["shift_id"]:
        return jsonify({"error": "No active shift. Please login first."}), 400

    try:
        sim_response = requests.get("http://127.0.0.1:5001/get_current_data")
        sim_response.raise_for_status()
        sensor_data = sim_response.json()
    except requests.RequestException:
        return jsonify({"error": "Could not connect to simulator."}), 500

    conn = get_db_connection()
    cur = conn.cursor()
    new_alerts = []

    try:
        if any(p < ALERT_THRESHOLDS["PROXIMITY_NEAR"] for p in sensor_data['safety']['proximity_meters'].values()):
            new_alerts.append({"type": "PROXIMITY_NEAR", "message": "Proximity Breach! Object too close."})
        if sensor_data['environment']['noise_db'] > ALERT_THRESHOLDS["HIGH_NOISE"]:
            new_alerts.append({"type": "HIGH_NOISE", "message": "Noise levels exceed safety threshold."})
        if sensor_data['environment']['dust_aqi'] > ALERT_THRESHOLDS["HIGH_AQI"]:
            new_alerts.append({"type": "HIGH_AQI", "message": "Air quality hazardous. High dust levels."})
        if sensor_data['status']['engine_temperature_celsius'] > ALERT_THRESHOLDS["HIGH_ENGINE_TEMP"]:
            new_alerts.append({"type": "HIGH_ENGINE_TEMP", "message": "Engine temperature critical!"})

        if active_shift["geofence_wkt"]:
            cur.execute(
                "SELECT NOT ST_Contains(ST_GeomFromText(%s, 4326), ST_SetSRID(ST_MakePoint(%s, %s), 4326));",
                (active_shift['geofence_wkt'], sensor_data['location']['gps']['longitude'], sensor_data['location']['gps']['latitude'])
            )
            if cur.fetchone()[0]:
                new_alerts.append({"type": "GEOFENCE_BREACH", "message": "Machine is outside designated work area."})

        for alert in new_alerts:
            cur.execute(
                "INSERT INTO events (shift_id, event_type, details) VALUES (%s, %s, %s);",
                (active_shift['shift_id'], alert['type'], psycopg2.extras.Json(alert))
            )
        conn.commit()
    except Exception as e:
        print(f"Database Error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    return jsonify({"live_data": sensor_data, "alerts": new_alerts})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)