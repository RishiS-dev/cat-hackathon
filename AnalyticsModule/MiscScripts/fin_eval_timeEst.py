import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

def evaluate_final_time_model(data_path='fin_synthetic_machine_data.csv', model_path='fin_task_duration_model.joblib'):
    """
    Loads the final saved task duration model and evaluates its performance
    on the final test set.
    """
    print(f"1. Loading final model from '{model_path}'...")
    try:
        # Construct absolute paths based on the script's location
        BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        model_pipeline = joblib.load(os.path.join(BASE_DIR, model_path))
    except FileNotFoundError:
        print(f"Error: The model file '{model_path}' was not found.")
        print("Please make sure you have trained the final model first.")
        return

    print(f"2. Loading and preparing final data from '{data_path}'...")
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, data_path))
    except FileNotFoundError:
        print(f"Error: The data file '{data_path}' was not found.")
        return
    
    # --- Recreate the exact same test set as during training ---
    # This logic MUST match the final training script exactly.
    print("3. Engineering the 'Task_Duration_Hours' target variable...")
    terrain_multipliers = df['Terrain'].map({'Flat': 1.0, 'Incline': 1.15, 'Steep': 1.30})
    df['Task_Duration_Hours'] = ((df['Idling_Time'] / 60) + (df['Load_Cycles'] * 0.1)) * terrain_multipliers
    df = df.dropna()

    # Define the features and target, matching the final training script
    target = 'Task_Duration_Hours'
    features = [
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

    X = df[features]
    y = df[target]
    
    # Use the same random_state to get the identical split as in training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("4. Making predictions on the final test data...")
    predictions = model_pipeline.predict(X_test)

    # 5. Calculate performance metrics
    print("\n--- Final Model Performance ---")
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"✅ Mean Absolute Error (MAE): {mae:.4f} hours")
    print(f"✅ Root Mean Squared Error (RMSE): {rmse:.4f} hours")
    print("-------------------------")
    print("(MAE means on average, the prediction is off by this many hours.)")

    # 6. Show a few examples side-by-side
    print("\n--- Sample Predictions vs. Actual Values (Final) ---")
    comparison_df = pd.DataFrame({
        'Actual Duration (Hrs)': y_test, 
        'Predicted Duration (Hrs)': predictions
    })
    comparison_df['Difference (Hrs)'] = comparison_df['Actual Duration (Hrs)'] - comparison_df['Predicted Duration (Hrs)']
    
    print(comparison_df.head(10).to_string())
    print("---------------------------------------------------\n")

if __name__ == "__main__":
    evaluate_final_time_model()
