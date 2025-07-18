# analytics.py
from flask import Blueprint, request, jsonify
import os
import pandas as pd
import numpy as np
import joblib

analytics_bp = Blueprint('analytics', __name__)

# --- Load Models and Data ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'fin_synthetic_machine_data.csv')

try:
    time_model = joblib.load(os.path.join(BASE_DIR, 'fin_task_duration_model.joblib'))
    profiler_model = joblib.load(os.path.join(BASE_DIR, 'fin_operator_profiler_model.joblib'))
    health_model = joblib.load(os.path.join(BASE_DIR, 'fin_machine_health_model.joblib'))
    df = pd.read_csv(DATA_PATH)

    # --- Preprocess for Profiler ---
    total_hours = df.groupby('Operator_ID')['Engine_Hours'].apply(lambda x: x.max() - x.min()).replace(0, 1)
    operator_profiles = df.groupby('Operator_ID').agg(
        total_load_cycles=('Load_Cycles', 'sum'),
        total_fuel_used=('Fuel_Used', 'sum'),
        total_idling_time_min=('Idling_Time', 'sum'),
        total_safety_alerts=('Safety_Alert_Triggered', lambda x: (x == 'Yes').sum())
    ).reset_index()
    operator_profiles = operator_profiles.merge(total_hours.rename('total_engine_hours'), on='Operator_ID')
    operator_profiles['fuel_per_load_cycle'] = operator_profiles['total_fuel_used'] / operator_profiles['total_load_cycles']
    operator_profiles['idling_ratio'] = operator_profiles['total_idling_time_min'] / 60 / operator_profiles['total_engine_hours']
    operator_profiles['safety_incident_rate'] = operator_profiles['total_safety_alerts'] / operator_profiles['total_engine_hours']
    features_for_clustering = ['fuel_per_load_cycle', 'idling_ratio', 'safety_incident_rate']
    X_profiles = operator_profiles[features_for_clustering].replace([np.inf, -np.inf], 0).fillna(0)
    operator_profiles['cluster'] = profiler_model.predict(X_profiles)

    print("✅ Analytics module: Models and data loaded successfully.")
except Exception as e:
    print(f"❌ Analytics module startup error: {e}")
    time_model, profiler_model, health_model = None, None, None
    operator_profiles, X_profiles = None, None

# --- Helper for Health Insights ---
def get_actionable_insight(row):
    if row['Fuel_Used'] > 15 and row['RPM'] < 900:
        return "High fuel use at low RPM. Check for potential fluid leaks."
    if row['Temperature_C'] > 140 and row['Load_Cycles'] < 5:
        return "Engine overheating under light load. Check radiator for debris."
    if row['RPM'] > 2000 and row['Load_Cycles'] == 0:
        return "High RPM with no load. Reduce throttle to save fuel."
    return "Anomaly detected. Perform system check."

# --- API Endpoints ---
@analytics_bp.route('/api/profiler_data')
def profiler_data():
    if profiler_model is None:
        return jsonify({"error": "Profiler model not available"}), 500

    scatter = profiler_model.named_steps['scaler'].transform(X_profiles)
    avg_fuel = operator_profiles['fuel_per_load_cycle'].mean()
    avg_idle = operator_profiles['idling_ratio'].mean()
    avg_safety = operator_profiles['safety_incident_rate'].mean()

    response = {
        'site_average': {
            'fuel_efficiency_score': (1 / avg_fuel if avg_fuel else 0) * 100,
            'low_idling_score': (1 - avg_idle) * 100,
            'safety_score': (1 - avg_safety) * 100
        },
        'operators': []
    }

    for i, row in operator_profiles.iterrows():
        response['operators'].append({
            'id': row['Operator_ID'],
            'cluster': int(row['cluster']),
            'scatter_x': scatter[i, 0],
            'scatter_y': scatter[i, 1],
            'fuel_efficiency_score': (1 / row['fuel_per_load_cycle'] if row['fuel_per_load_cycle'] else 0) * 100,
            'low_idling_score': (1 - row['idling_ratio']) * 100,
            'safety_score': (1 - row['safety_incident_rate']) * 100
        })
    return jsonify(response)

@analytics_bp.route('/api/estimate_time', methods=['POST'])
def estimate_time():
    if time_model is None:
        return jsonify({"error": "Task duration model not available"}), 500

    data = request.get_json()
    features = ['Machine_ID', 'Operator_ID', 'RPM', 'Task_Type', 'Soil_Type', 'Terrain',
                'Load_Cycles', 'Temperature_C', 'Precipitation_mm']
    input_df = pd.DataFrame([data], columns=features)
    prediction = time_model.predict(input_df)[0]
    return jsonify({
        "predicted_duration_hours": float(prediction),
        "readable_duration": f"{int(prediction)} hours and {int((prediction * 60) % 60)} minutes"
    })

@analytics_bp.route('/api/check_health', methods=['POST'])
def check_health():
    if health_model is None:
        return jsonify({"error": "Machine health model not available"}), 500

    data = request.get_json()
    features = ['RPM', 'Engine_Hours', 'Fuel_Used', 'Load_Cycles', 'Idling_Time', 'Temperature_C', 'Precipitation_mm']
    input_df = pd.DataFrame([data], columns=features)
    score = health_model.decision_function(input_df)[0]

    ANOMALY_THRESHOLD = -0.07
    is_anomaly = score < ANOMALY_THRESHOLD
    insight = get_actionable_insight(data) if is_anomaly else ""

    return jsonify({
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": float(score),
        "actionable_insight": insight
    })
