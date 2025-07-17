from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app)

# --- Load Models and Data ONCE at startup ---
try:
    df = pd.read_csv('synthetic_machine_data_v3.csv')
    profiler_model = joblib.load('operator_profiler_model.joblib')
    print("✅ Models and data loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}. Make sure all .joblib and .csv files are in the same directory.")
    exit()

# --- Pre-process Profiler Data ---
print("Pre-processing operator profiles...")
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
print("✅ Operator profiles ready.")


# --- API Endpoints ---
@app.route('/')
def index():
    return "<h1>Analytics Server is Running!</h1>"

@app.route('/api/profiler_data')
def get_profiler_data():
    """
    API endpoint to provide all data needed for all three charts.
    """
    scatter_data = profiler_model.named_steps['scaler'].transform(X_profiles)
    
    # Calculate site averages for comparison
    avg_fuel_per_cycle = operator_profiles['fuel_per_load_cycle'].mean()
    avg_idling_ratio = operator_profiles['idling_ratio'].mean()
    avg_safety_rate = operator_profiles['safety_incident_rate'].mean()

    response_data = {
        'operators': [],
        'site_average': {
            'fuel_efficiency_score': (1 / avg_fuel_per_cycle if avg_fuel_per_cycle != 0 else 0) * 100,
            'low_idling_score': (1 - avg_idling_ratio) * 100,
            'safety_score': (1 - avg_safety_rate) * 100
        }
    }

    for i, row in operator_profiles.iterrows():
        response_data['operators'].append({
            'id': row['Operator_ID'],
            'cluster': int(row['cluster']),
            'scatter_x': scatter_data[i, 0],
            'scatter_y': scatter_data[i, 1],
            # Data for Radar and Bar charts (scaled 0-100)
            'fuel_efficiency_score': (1 / row['fuel_per_load_cycle'] if row['fuel_per_load_cycle'] != 0 else 0) * 100,
            'low_idling_score': (1 - row['idling_ratio']) * 100,
            'safety_score': (1 - row['safety_incident_rate']) * 100
        })
        
    return jsonify(response_data)


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)
