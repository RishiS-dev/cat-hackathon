import pandas as pd
import numpy as np
import joblib
import os

def get_actionable_insight(data_row):
    """
    Analyzes a single anomalous data row and returns a specific,
    actionable insight for the operator. This is the "recommendation engine".
    """
    # Rule 1: High fuel consumption while idling
    if data_row['Fuel_Used'] > 15 and data_row['RPM'] < 900:
        return "💡 INSIGHT: High fuel use at low RPM detected. Check for potential fluid leaks."

    # Rule 2: Overheating under light load
    if data_row['Temperature_C'] > 140 and data_row['Load_Cycles'] < 5:
        return "💡 INSIGHT: Engine overheating under light load. Check radiator for debris blockage."
        
    # Rule 3: High RPM with no work being done (inefficient operation)
    if data_row['RPM'] > 2000 and data_row['Load_Cycles'] == 0:
        return "💡 INSIGHT: High engine speed with no load. Avoid excessive throttle when not actively working to save fuel."

    # Default catch-all message if no specific rule matches
    return "💡 INSIGHT: General machine health anomaly detected. Recommend a standard systems check."


def evaluate_final_health_model(model_path='fin_machine_health_model.joblib'):
    """
    Loads the saved machine health model and tests it against a mix of
    normal and hand-crafted anomalous data points, then generates actionable insights.
    """
    print(f"1. Loading final machine health model from '{model_path}'...")
    try:
        BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        model = joblib.load(os.path.join(BASE_DIR, model_path))
    except FileNotFoundError:
        print(f"Error: The model file '{model_path}' was not found.")
        return

    # --- Step 1: Create Comprehensive Test Data ---
    print("\n2. Creating test cases with normal and anomalous data...")
    test_cases = [
        # Normal operations
        {'case': 'Normal Hauling', 'data': {'RPM': 1850, 'Engine_Hours': 1500, 'Fuel_Used': 11.0, 'Load_Cycles': 12, 'Idling_Time': 25, 'Temperature_C': 102, 'Precipitation_mm': 0}},
        {'case': 'Normal Idling', 'data': {'RPM': 800, 'Engine_Hours': 1502, 'Fuel_Used': 2.5, 'Load_Cycles': 0, 'Idling_Time': 60, 'Temperature_C': 95, 'Precipitation_mm': 0}},
        
        # Actionable Anomalies
        {'case': 'High Fuel at Idle', 'data': {'RPM': 850, 'Engine_Hours': 1800, 'Fuel_Used': 18.0, 'Load_Cycles': 0, 'Idling_Time': 60, 'Temperature_C': 98, 'Precipitation_mm': 0}},
        {'case': 'Overheating at Idle', 'data': {'RPM': 820, 'Engine_Hours': 1200, 'Fuel_Used': 4.0, 'Load_Cycles': 1, 'Idling_Time': 50, 'Temperature_C': 150, 'Precipitation_mm': 0}},
        {'case': 'Inefficient Revving', 'data': {'RPM': 2300, 'Engine_Hours': 1400, 'Fuel_Used': 9.0, 'Load_Cycles': 0, 'Idling_Time': 10, 'Temperature_C': 99, 'Precipitation_mm': 0}},
    ]
    
    test_df = pd.DataFrame([case['data'] for case in test_cases])
    test_df['case_description'] = [case['case'] for case in test_cases]
    
    print("   - Test data created successfully.")

    # --- Step 2: Get Anomaly Scores (The Fix is Here) ---
    print("\n3. Using the model to get an 'anomaly score' for each data point...")
    # Instead of a simple predict (-1 or 1), we get a continuous score.
    # The more negative the score, the more anomalous the data is.
    scores = model.decision_function(test_df.drop(columns=['case_description']))
    test_df['anomaly_score'] = scores
    
    # REVISED: Define a more sensitive threshold to catch the anomalies.
    ANOMALY_THRESHOLD = -0.04
    
    # Classify based on our custom threshold.
    test_df['is_anomaly'] = np.where(test_df['anomaly_score'] < ANOMALY_THRESHOLD, 'ANOMALY', 'Normal')

    # --- Step 3: Generate Insights and Display Final Report ---
    print(f"\n--- Final Machine Health Evaluation Report (Threshold: {ANOMALY_THRESHOLD}) ---")
    for index, row in test_df.iterrows():
        print(f"\n--- Test Case: {row['case_description']} ---")
        print(f"Anomaly Score: {row['anomaly_score']:.4f}")
        print(f"Model Prediction: {row['is_anomaly']}")
        
        # If an anomaly is detected based on our threshold, generate the insight.
        if row['is_anomaly'] == 'ANOMALY':
            insight = get_actionable_insight(row)
            print(insight)
        
    print("\n--------------------------------------------")

if __name__ == "__main__":
    evaluate_final_health_model()
