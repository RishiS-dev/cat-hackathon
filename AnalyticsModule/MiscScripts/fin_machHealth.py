import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_final_health_model(data_path='fin_synthetic_machine_data.csv'):
    """
    Loads the final machine data and trains an Isolation Forest model 
    to detect anomalies in sensor readings for machine health monitoring.
    """
    print(f"1. Loading final data from '{data_path}'...")
    try:
        # Construct absolute path based on the script's location
        BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        df = pd.read_csv(os.path.join(BASE_DIR, data_path))
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        print("Please make sure you have generated the final dataset first.")
        return

    # --- Step 1: Feature Selection ---
    # We select the features that represent a machine's live sensor state.
    # The new RPM feature is crucial for detecting complex anomalies.
    print("2. Selecting features for machine health monitoring...")
    
    features_for_health = [
        'RPM',
        'Engine_Hours',
        'Fuel_Used',
        'Load_Cycles',
        'Idling_Time',
        'Temperature_C',
        'Precipitation_mm'
    ]
    
    X = df[features_for_health].fillna(0)
    
    print("   - Features selected successfully.")
    print(X.head())

    # --- Step 2: Training the Isolation Forest Model ---
    # This model learns the boundaries of "normal" data.
    print("\n3. Training the final Isolation Forest model...")

    # 'contamination' tells the model what percentage of the data it should
    # consider to be anomalies. 'auto' is a safe and modern default.
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)

    model.fit(X)

    # --- Step 3: Save the Trained Model ---
    model_filename = 'fin_machine_health_model.joblib'
    joblib.dump(model, os.path.join(BASE_DIR, model_filename))
    print(f"\n4. Final machine health model saved successfully as '{model_filename}'")

    # --- Step 4: Demonstrate by Finding Anomalies in Training Data ---
    # This shows the model is working by finding the most unusual data points
    # within the data it was just trained on.
    print("\n--- Anomaly Detection Demonstration ---")
    
    scores = model.decision_function(X)
    df['anomaly_score'] = scores
    
    # Sort by the score to see the top 5 most anomalous rows
    top_anomalies = df.sort_values('anomaly_score').head(5)
    
    print("Top 5 most anomalous data points found in the training data:")
    print(top_anomalies[features_for_health].to_string())
    print("\nThese are the types of 'weird' data points the model has learned to flag.")


if __name__ == "__main__":
    train_final_health_model()
