import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

def train_health_model(data_path='synthetic_machine_data_v3.csv'):
    """
    Loads machine data and trains an Isolation Forest model to detect
    anomalies in sensor readings, representing machine health issues.
    """
    print(f"1. Loading data from '{data_path}'...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        print("Please make sure you have generated the v3 dataset first.")
        return

    # --- Step 1: Feature Selection ---
    # We select only the features that represent a machine's live
    # sensor state at a single point in time. Contextual data like
    # Operator_ID or Task_Type is excluded.
    print("2. Selecting features for machine health monitoring...")
    
    features_for_health = [
        'Engine_Hours',
        'Fuel_Used',
        'Load_Cycles',
        'Idling_Time',
        'Temperature_C',
        'Precipitation_mm' # Weather can affect normal operation
    ]
    
    X = df[features_for_health].fillna(0)
    
    print("   - Features selected successfully.")
    print(X.head())

    # --- Step 2: Training the Isolation Forest Model ---
    # This model learns the boundaries of "normal" data.
    print("\n3. Training the Isolation Forest model...")

    # 'contamination' tells the model what percentage of the data it should
    # consider to be anomalies. 'auto' is a safe and modern default.
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)

    model.fit(X)

    # --- Step 3: Save the Trained Model ---
    model_filename = 'machine_health_model.joblib'
    joblib.dump(model, model_filename)
    print(f"\n4. Machine health model saved successfully as '{model_filename}'")

    # --- Step 4: Demonstrate by Finding Anomalies in Training Data ---
    # To show that it works, we can use the model to find the most
    # anomalous data points within the data it was trained on.
    print("\n--- Anomaly Detection Demonstration ---")
    
    # .decision_function gives a score; the lower the score, the more anomalous.
    scores = model.decision_function(X)
    
    # Add scores to the original dataframe to see which rows are most anomalous
    df['anomaly_score'] = scores
    
    # Sort by the score to see the top 5 most anomalous rows
    top_anomalies = df.sort_values('anomaly_score').head(5)
    
    print("Top 5 most anomalous data points found in the training data:")
    print(top_anomalies[features_for_health].to_string())
    print("\nThese are the types of 'weird' data points the model has learned to flag.")


if __name__ == "__main__":
    train_health_model()
