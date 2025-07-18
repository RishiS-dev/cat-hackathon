import flask
from flask import request, jsonify
import joblib
import pandas as pd
import numpy as np
import traceback
import os

# -------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------

# Initialize the Flask application
app = flask.Flask(__name__)

# --- Model & Column Information ---

# Get the absolute path of the directory where this script is located.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to your trained machine health model.
MODEL_PATH = os.path.join(SCRIPT_DIR, 'fin_machine_health_model.joblib')

# Define the exact feature columns the model was trained on, in the correct order.
MODEL_FEATURES = [
    'RPM',
    'Engine_Hours',
    'Fuel_Used',
    'Load_Cycles',
    'Idling_Time',
    'Temperature_C',
    'Precipitation_mm'
]

# Define the threshold for classifying a data point as an anomaly.
# This value is taken from your evaluation script.
ANOMALY_THRESHOLD = -0.04

# --- Load Model ---
model = None
try:
    print(f"Loading model from: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Machine health model loaded successfully.")
    else:
        print(f"Error: Model file not found at the specified path.")
        print("Please ensure 'fin_machine_health_model.joblib' is in the same directory as this script.")

except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    traceback.print_exc()

# -------------------------------------------------------------------
# Actionable Insights Logic (from your evaluation script)
# -------------------------------------------------------------------

def get_actionable_insight(data_row):
    """
    Analyzes a single anomalous data row and returns a specific,
    actionable insight for the operator. This is the "recommendation engine".
    
    Args:
        data_row (pd.Series): A pandas Series containing the feature values for the anomaly.
    
    Returns:
        str: A string containing the actionable insight.
    """
    # Rule 1: High fuel consumption while idling
    if data_row['Fuel_Used'] > 15 and data_row['RPM'] < 900:
        return "High fuel use at low RPM detected. Advise operator to check for potential fluid leaks."

    # Rule 2: Overheating under light load
    if data_row['Temperature_C'] > 140 and data_row['Load_Cycles'] < 5:
        return "Engine overheating under light load. Advise operator to check radiator for debris blockage."
        
    # Rule 3: High RPM with no work being done (inefficient operation)
    if data_row['RPM'] > 2000 and data_row['Load_Cycles'] == 0:
        return "High engine speed with no load. Advise operator to avoid excessive throttle when not actively working to save fuel."

    # Default catch-all message if no specific rule matches
    return "General machine health anomaly detected. Recommend a standard systems check."

# -------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------

@app.route("/")
def index():
    """A simple endpoint to check if the API is running."""
    return "Welcome to the Machine Health Monitoring API! Use the /predict/machine_health endpoint for predictions."

@app.route("/predict/machine_health", methods=['POST'])
def predict_machine_health():
    """
    Endpoint to predict machine health status and provide actionable insights.
    Accepts a JSON payload with live sensor data.
    """
    if model is None:
        return jsonify({
            "error": "Model is not loaded. The server could not start correctly. Please check server logs."
        }), 500

    json_data = request.get_json()

    if not json_data:
        return jsonify({"error": "No input data provided. Please POST a JSON object."}), 400

    missing_keys = [key for key in MODEL_FEATURES if key not in json_data]
    if missing_keys:
        return jsonify({
            "error": "Missing required features in JSON payload.",
            "missing_keys": missing_keys
        }), 400

    try:
        # --- Data Preparation ---
        input_data = {key: [json_data[key]] for key in MODEL_FEATURES}
        input_df = pd.DataFrame.from_dict(input_data)
        print(f"Received data for health check:\n{input_df.to_string()}")

        # --- Prediction & Analysis ---
        # Get the raw anomaly score from the Isolation Forest model.
        score = model.decision_function(input_df)[0]
        status = "Normal"
        insight = "All systems operating within normal parameters."

        # Check if the score is below our defined threshold.
        if score < ANOMALY_THRESHOLD:
            status = "ANOMALY"
            # If it's an anomaly, get the specific actionable insight.
            # We pass the first (and only) row of the DataFrame to the function.
            insight = get_actionable_insight(input_df.iloc[0])

        # --- Response ---
        response = {
            "success": True,
            "status": status,
            "anomaly_score": round(float(score), 4),
            "actionable_insight": insight,
            "input_features": json_data
        }
        return jsonify(response), 200

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "error": "An error occurred during prediction.",
            "message": str(e)
        }), 500

# -------------------------------------------------------------------
# Main execution block
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Run on a different port (e.g., 5002) to avoid conflict with the task duration API.
    app.run(host='0.0.0.0', port=5002, debug=True)
