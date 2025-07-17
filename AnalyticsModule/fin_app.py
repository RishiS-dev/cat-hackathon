import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import os

# --- Configuration ---
# This makes the script runnable from any location by finding its own path.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'fin_synthetic_machine_data.csv')

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Load All Models and Data at Startup ---
print("--- Initializing Analytics Server ---")
try:
    print("1. Loading all final models...")
    time_model = joblib.load(os.path.join(BASE_DIR, 'fin_task_duration_model.joblib'))
    profiler_model = joblib.load(os.path.join(BASE_DIR, 'fin_operator_profiler_model.joblib'))
    health_model = joblib.load(os.path.join(BASE_DIR, 'fin_machine_health_model.joblib'))
    print("   - All models loaded successfully.")

    print("2. Loading and pre-processing data...")
    df = pd.read_csv(DATA_PATH)
    
    # Pre-process data required for the profiler
    total_hours = df.groupby('Operator_ID')['Engine_Hours'].apply(lambda x: x.max() - x.min()).replace(0, 1)
    operator_profiles = df.groupby('Operator_ID').agg(
        total_load_cycles=('Load_Cycles', 'sum'),
        total_fuel_used=('Fuel_Used', 'sum'),
        total_idling_time_min=('Idling_Time', 'sum'),
        total_safety_alerts=('Safety_Alert_Triggered', lambda x: (x == 'Yes').sum())
    ).reset_index()
    operator_profiles = operator_profiles.merge(total_hours.rename('total_engine_hours'), on='Operator_ID')
    operator_profiles['fuel_per_load_cycle'] = (operator_profiles['total_fuel_used'] / operator_profiles['total_load_cycles']).replace([np.inf, -np.inf], 0)
    operator_profiles['idling_ratio'] = (operator_profiles['total_idling_time_min'] / 60 / operator_profiles['total_engine_hours']).replace([np.inf, -np.inf], 0)
    operator_profiles['safety_incident_rate'] = (operator_profiles['total_safety_alerts'] / operator_profiles['total_engine_hours']).replace([np.inf, -np.inf], 0)
    features_for_clustering = ['fuel_per_load_cycle', 'idling_ratio', 'safety_incident_rate']
    X_profiles = operator_profiles[features_for_clustering].fillna(0)
    operator_profiles['cluster'] = profiler_model.predict(X_profiles)
    print("   - Operator profiles ready.")

except FileNotFoundError as e:
    print(f"❌ CRITICAL ERROR: Could not load a required file: {e}")
    print("   - Please ensure all 'fin_*.csv' and 'fin_*.joblib' files are in the same directory as this script.")
    exit()

# --- Helper function for actionable insights ---
def get_actionable_insight(data_row):
    if data_row['Fuel_Used'] > 15 and data_row['RPM'] < 900:
        return "High fuel use at low RPM detected. Check for potential fluid leaks."
    if data_row['Temperature_C'] > 140 and data_row['Load_Cycles'] < 5:
        return "Engine overheating under light load. Check radiator for debris blockage."
    if data_row['RPM'] > 2000 and data_row['Load_Cycles'] == 0:
        return "High engine speed with no load. Avoid excessive throttle when not actively working to save fuel."
    return "General machine health anomaly detected. Recommend a standard systems check."

# --- API Endpoints ---

@app.route('/')
def index():
    return "<h1>CatHackathon Analytics Server is Running!</h1>"

@app.route('/api/profiler_data')
def get_profiler_data():
    """Endpoint for the main analytics dashboard charts."""
    # (Code for this endpoint is the same as the previous final version)
    scatter_data = profiler_model.named_steps['scaler'].transform(X_profiles)
    avg_fuel_per_cycle = operator_profiles['fuel_per_load_cycle'].mean()
    avg_idling_ratio = operator_profiles['idling_ratio'].mean()
    avg_safety_rate = operator_profiles['safety_incident_rate'].mean()
    response_data = {'operators': [], 'site_average': {
        'fuel_efficiency_score': (1 / avg_fuel_per_cycle if avg_fuel_per_cycle != 0 else 0) * 100,
        'low_idling_score': (1 - avg_idling_ratio) * 100,
        'safety_score': (1 - avg_safety_rate) * 100 }}
    for i, row in operator_profiles.iterrows():
        response_data['operators'].append({
            'id': row['Operator_ID'], 'cluster': int(row['cluster']),
            'scatter_x': scatter_data[i, 0], 'scatter_y': scatter_data[i, 1],
            'fuel_efficiency_score': (1 / row['fuel_per_load_cycle'] if row['fuel_per_load_cycle'] != 0 else 0) * 100,
            'low_idling_score': (1 - row['idling_ratio']) * 100,
            'safety_score': (1 - row['safety_incident_rate']) * 100 })
    return jsonify(response_data)

@app.route('/api/estimate_time', methods=['POST'])
def estimate_time():
    """
    NEW: Endpoint to predict task completion time.
    Expects a JSON payload with task details.
    """
    data = request.get_json()
    # Create a DataFrame in the exact order the model expects
    features = ['Machine_ID', 'Operator_ID', 'RPM', 'Task_Type', 'Soil_Type', 'Terrain', 'Load_Cycles', 'Temperature_C', 'Precipitation_mm']
    input_df = pd.DataFrame([data], columns=features)
    
    prediction = time_model.predict(input_df)
    
    # Convert prediction to hours and minutes for readability
    hours = int(prediction[0])
    minutes = int((prediction[0] * 60) % 60)
    
    return jsonify({
        'predicted_duration_hours': prediction[0],
        'readable_duration': f"{hours} hours and {minutes} minutes"
    })

@app.route('/api/check_health', methods=['POST'])
def check_health():
    """
    NEW: Endpoint to check the health of the machine based on live sensor data.
    """
    data = request.get_json()
    # Create a DataFrame in the exact order the model expects
    features = ['RPM', 'Engine_Hours', 'Fuel_Used', 'Load_Cycles', 'Idling_Time', 'Temperature_C', 'Precipitation_mm']
    input_df = pd.DataFrame([data], columns=features)
    
    # Get the raw anomaly score from the model
    score = health_model.decision_function(input_df)[0]
    
    # Use our custom threshold to decide if it's an anomaly
    ANOMALY_THRESHOLD = -0.07
    is_anomaly = score < ANOMALY_THRESHOLD
    
    insight = ""
    if is_anomaly:
        insight = get_actionable_insight(data)
        
    return jsonify({
        'is_anomaly': bool(is_anomaly),
        'anomaly_score': float(score),
        'actionable_insight': insight
    })

# --- Run the App ---
if __name__ == '__main__':
    print("--- Server is ready and running at http://127.0.0.1:5001 ---")
    app.run(debug=True, port=5001)
