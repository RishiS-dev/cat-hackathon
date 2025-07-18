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
# This makes the script more robust and independent of the current working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to your trained model, joining the script's directory with the model filename.
# This ensures the model is found as long as it's in the same directory as this API script.
MODEL_PATH = os.path.join(SCRIPT_DIR, 'fin_task_duration_model.joblib')


# Define the exact feature columns the model was trained on, in the correct order.
# This is crucial for the model to make accurate predictions.
MODEL_FEATURES = [
    'Machine_ID',
    'Operator_ID',
    'RPM',
    'Task_Type',
    'Soil_Type',
    'Terrain',
    'Load_Cycles',
    'Temperature_C',
    'Precipitation_mm'
]

# --- Load Model ---
# Load the trained model pipeline once when the server starts.
# This is more efficient than loading it for every single request.
model_pipeline = None
try:
    print(f"Loading model from: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at the specified path.")
        print("Please ensure 'fin_task_duration_model.joblib' is in the same directory as this script.")

except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    traceback.print_exc()


# -------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------

@app.route("/")
def index():
    """A simple endpoint to check if the API is running."""
    return "Welcome to the Caterpillar Analytics API! Use the /predict/task_duration endpoint for predictions."

@app.route("/predict/task_duration", methods=['POST'])
def predict_task_duration():
    """
    Endpoint to predict the duration of a task.
    Accepts a JSON payload with the required features.
    """
    # Ensure the model was loaded correctly before proceeding
    if model_pipeline is None:
        return jsonify({
            "error": "Model is not loaded. The server could not start correctly. Please check server logs."
        }), 500

    # Get the JSON data from the POST request
    json_data = request.get_json()

    # --- Input Validation ---
    if not json_data:
        return jsonify({"error": "No input data provided. Please POST a JSON object."}), 400

    # Check if all required features are in the JSON data
    missing_keys = [key for key in MODEL_FEATURES if key not in json_data]
    if missing_keys:
        return jsonify({
            "error": "Missing required features in JSON payload.",
            "missing_keys": missing_keys
        }), 400

    try:
        # --- Data Preparation ---
        # Convert the incoming JSON into a pandas DataFrame.
        # The model's pipeline expects a DataFrame as input.
        # We use a dictionary comprehension to ensure the order of columns matches MODEL_FEATURES.
        input_data = {key: [json_data[key]] for key in MODEL_FEATURES}
        input_df = pd.DataFrame.from_dict(input_data)

        print(f"Received data for prediction:\n{input_df.to_string()}")

        # --- Prediction ---
        # The .predict() method uses the entire pipeline (preprocessing + regressor)
        prediction_array = model_pipeline.predict(input_df)

        # The prediction is a numpy array, e.g., array([5.432]). We extract the single value.
        predicted_duration_hours = float(prediction_array[0])

        # --- Response ---
        # Create a success response payload
        response = {
            "success": True,
            "predicted_duration_hours": round(predicted_duration_hours, 4),
            "input_features": json_data
        }
        return jsonify(response), 200

    except Exception as e:
        # --- Error Handling ---
        # Provide a detailed error message for debugging
        print(traceback.format_exc())
        return jsonify({
            "error": "An error occurred during prediction.",
            "message": str(e)
        }), 500

# -------------------------------------------------------------------
# Main execution block
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Set host to '0.0.0.0' to make the API accessible from other machines on the network.
    # Use a standard port like 5001 to avoid conflicts with other services.
    app.run(host='0.0.0.0', port=5002, debug=True)
