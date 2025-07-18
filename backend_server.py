import os
import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- Initialization ---
load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

active_shift = { "shift_id": None, "task_id": None, "geofence_wkt": None }

# --- API Endpoints Configuration ---
ML_API_ENDPOINT = "http://127.0.0.1:5002/predict/task_duration" # ML model runs on port 5002
SIMULATOR_API_ENDPOINT = "http://127.0.0.1:5001/get_current_data"

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
    "PROXIMITY_NEAR": 3.0, "HIGH_NOISE": 90.0,
    "HIGH_AQI": 200.0, "HIGH_ENGINE_TEMP": 115.0
}

# --- API Endpoints ---
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

@app.route('/api/predict_time', methods=['GET'])
def predict_time():
    task_id = request.args.get('task_id', type=int)
    if not task_id: return jsonify({"error": "task_id is required"}), 400
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM scheduled_tasks WHERE task_id = %s;", (task_id,))
    task_data = cur.fetchone()
    cur.close()
    conn.close()

    if not task_data: return jsonify({"error": "Task not found"}), 404

    # Construct the payload for the ML model API
    task_inputs = task_data['task_inputs']
    ml_payload = {
        "Machine_ID": task_data['machine_id'],
        "Operator_ID": task_data['operator_id'],
        "RPM": task_inputs.get('average_rpm', 1800), # Use a default if not provided
        "Task_Type": task_data['task_type'],
        "Soil_Type": task_inputs.get('soil_type'),
        "Terrain": task_inputs.get('terrain'),
        "Load_Cycles": task_data['load_cycles_planned'],
        "Temperature_C": task_inputs.get('temperature_c'),
        "Precipitation_mm": task_inputs.get('precipitation_mm')
    }

    try:
        # Call the external ML model API
        ml_response = requests.post(ML_API_ENDPOINT, json=ml_payload, timeout=5)
        ml_response.raise_for_status()
        prediction_data = ml_response.json()
        
        # Return the prediction to the frontend
        return jsonify({
            "task_id": task_id,
            "predicted_duration_hours": prediction_data.get('predicted_duration_hours')
        })
    except requests.RequestException as e:
        print(f"Error calling ML API: {e}")
        return jsonify({"error": "Failed to get prediction from ML model."}), 503

@app.route('/api/live_status', methods=['GET'])
def get_live_status():
    # ... (This function remains the same as before)
    if not active_shift["shift_id"]:
        return jsonify({"error": "No active shift. Please login first."}), 400
    try:
        sim_response = requests.get(SIMULATOR_API_ENDPOINT)
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
        if active_shift["geofence_wkt"]:
            cur.execute(
                "SELECT NOT ST_Contains(ST_GeomFromText(%s, 4326), ST_SetSRID(ST_MakePoint(%s, %s), 4326));",
                (active_shift['geofence_wkt'], sensor_data['location']['gps']['longitude'], sensor_data['location']['gps']['latitude'])
            )
            if cur.fetchone()[0]:
                new_alerts.append({"type": "GEOFENCE_BREACH", "message": "Machine is outside designated work area."})
        for alert in new_alerts:
            cur.execute("INSERT INTO events (shift_id, event_type, details) VALUES (%s, %s, %s);", (active_shift['shift_id'], alert['type'], psycopg2.extras.Json(alert)))
        conn.commit()
    except Exception as e:
        print(f"Database Error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
    return jsonify({"live_data": sensor_data, "alerts": new_alerts})

@app.route('/api/set_task', methods=['POST'])
def set_task():
    # ... (This function remains the same as before)
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

@app.route('/api/analytics/operator_summary', methods=['GET'])
def get_operator_summary():
    """
    Queries the database to get a summary of events for each operator.
    """
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # This SQL query joins the tables and counts events for each operator
    query = """
        SELECT
            op.operator_id,
            op.name,
            COUNT(ev.event_id) AS total_alerts,
            SUM(CASE WHEN ev.event_type = 'PROXIMITY_NEAR' THEN 1 ELSE 0 END) AS proximity_alerts,
            SUM(CASE WHEN ev.event_type = 'GEOFENCE_BREACH' THEN 1 ELSE 0 END) AS geofence_breaches
        FROM operators op
        LEFT JOIN work_shifts ws ON op.operator_id = ws.operator_id
        LEFT JOIN events ev ON ws.shift_id = ev.shift_id
        GROUP BY op.operator_id, op.name
        ORDER BY total_alerts DESC;
    """
    
    cur.execute(query)
    summary_data = [dict(row) for row in cur.fetchall()]
    
    cur.close()
    conn.close()
    
    return jsonify(summary_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
