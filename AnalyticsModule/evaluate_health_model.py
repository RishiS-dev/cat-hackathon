import pandas as pd
import numpy as np
import joblib

def evaluate_health_model(model_path='machine_health_model.joblib'):
    """
    Loads the saved machine health model and tests it against a mix of
    normal and hand-crafted anomalous data points to verify its performance.
    """
    print(f"1. Loading machine health model from '{model_path}'...")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: The model file '{model_path}' was not found.")
        print("Please make sure you have trained the health model first.")
        return

    # --- Step 1: Create Test Data ---
    # We will create a few examples of 'normal' data and 'bad' (anomalous) data.
    print("\n2. Creating test cases with normal and anomalous data...")

    # REVISED: These data points are now more representative of the "average"
    # machine operation from our v3 dataset to ensure they are not on the edge.
    normal_data = [
        # A very average "Hauling Sand" task
        {'Engine_Hours': 1500, 'Fuel_Used': 9.0, 'Load_Cycles': 10, 'Idling_Time': 30, 'Temperature_C': 95, 'Precipitation_mm': 0},
        # A very average "Trenching Loam" task
        {'Engine_Hours': 2100, 'Fuel_Used': 10.5, 'Load_Cycles': 12, 'Idling_Time': 25, 'Temperature_C': 105, 'Precipitation_mm': 0},
    ]

    # These data points are purposely "bad" to simulate machine health issues.
    anomalous_data = [
        # Anomaly 1: Very high fuel use while idling (potential leak/engine problem)
        {'Engine_Hours': 1800, 'Fuel_Used': 18.0, 'Load_Cycles': 0, 'Idling_Time': 60, 'Temperature_C': 98, 'Precipitation_mm': 0},
        
        # Anomaly 2: Overheating under no load (potential cooling system failure)
        {'Engine_Hours': 1200, 'Fuel_Used': 4.0, 'Load_Cycles': 1, 'Idling_Time': 50, 'Temperature_C': 150, 'Precipitation_mm': 0},
    ]

    test_df = pd.DataFrame(normal_data + anomalous_data)
    
    print("   - Test data created successfully.")
    print("Test Data:")
    print(test_df.to_string())

    # --- Step 2: Make Predictions ---
    # The model will predict -1 for anomalies and 1 for normal data points.
    print("\n3. Using the model to predict which data points are anomalous...")
    
    predictions = model.predict(test_df)
    
    # Add the predictions to our test dataframe for easy interpretation
    test_df['prediction'] = predictions
    test_df['is_anomaly'] = np.where(test_df['prediction'] == -1, 'Yes (Anomaly)', 'No (Normal)')

    # --- Step 3: Display and Verify Results ---
    print("\n--- Model Evaluation Results ---")
    print("The model should flag the 'bad' data as anomalies and the 'normal' data as not.")
    
    print("\nFinal Predictions:")
    print(test_df[['Fuel_Used', 'Load_Cycles', 'Temperature_C', 'is_anomaly']].to_string())
    
    print("\n--- Verification ---")
    # Check if the model correctly identified our hand-crafted anomalies
    num_anomalies_found = (test_df.iloc[2:]['prediction'] == -1).sum()
    num_normal_flagged = (test_df.iloc[:2]['prediction'] == -1).sum()

    if num_anomalies_found == len(anomalous_data) and num_normal_flagged == 0:
        print("✅ Success! The model correctly identified all hand-crafted anomalies and did not flag any normal data.")
    else:
        print("⚠️ Partial Success/Failure. Review the predictions above to see what the model missed.")


if __name__ == "__main__":
    evaluate_health_model()
