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

# --- Database Configuration ---
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"), "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# --- ML Model Simulation (Updated) ---
def simulate_ml_prediction(task_data):
    """Fake ML model with updated features."""
    base_time_per_cycle = 1.5 # minutes
    
    # Operator Persona Factor (instead of experience years)
    operator_factors = {"OP1001": 1.0, "OP1002": 0.9, "OP1003": 1.2, "OP1004": 0.85, "OP1005": 1.1}
    operator_factor = operator_factors.get(task_data['operator_id'], 1.0)

    # Categorical Factors
    soil_factors = {"Clay": 1.2, "Rock": 1.5, "Sand": 0.9, "Loam": 1.0, "Gravel": 1.1}
    terrain_factors = {"Steep": 1.3, "Incline": 1.1, "Flat": 1.0}
    
    task_inputs = task_data['task_inputs']
    soil_factor = soil_factors.get(task_inputs.get('soil_type'), 1.0)
    terrain_factor = terrain_factors.get(task_inputs.get('terrain'), 1.0)
    
    # Assume RPM is an average planned RPM for the task, passed in task_inputs
    avg_rpm = task_inputs.get('average_rpm', 1800)
    rpm_factor = 1800 / avg_rpm # Higher RPM = faster work (simplistic model)

    estimated_time = (
        task_data['load_cycles_planned'] * base_time_per_cycle * operator_factor * soil_factor * terrain_factor * rpm_factor
    )
    return round(estimated_time)

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

    estimated_minutes = simulate_ml_prediction(task_data)
    return jsonify({"task_id": task_id, "estimated_minutes": estimated_minutes})


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
    new_alerts = [] # Placeholder for alert logic
    try:
        cur.execute(
            """INSERT INTO sensor_logs (shift_id, location, engine_rpm) 
               VALUES (%s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s);""",
            (
                active_shift['shift_id'], sensor_data['location']['gps']['longitude'], 
                sensor_data['location']['gps']['latitude'], sensor_data['status']['engine_rpm']
            )
        )
        conn.commit()
    except Exception as e:
        print(f"Error during logging: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    return jsonify({"live_data": sensor_data, "alerts": new_alerts})

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)