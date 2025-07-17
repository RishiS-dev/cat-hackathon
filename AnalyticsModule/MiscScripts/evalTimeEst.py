import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_time_model_v3(data_path='synthetic_machine_data_v3.csv', model_path='task_duration_model_v3.joblib'):
    """
    Loads the saved v3 task duration model and evaluates its performance
    on the v3 test set.
    """
    print(f"1. Loading model from '{model_path}'...")
    try:
        model_pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: The model file '{model_path}' was not found.")
        print("Please make sure you have trained the v3 model first.")
        return

    print(f"2. Loading and preparing data from '{data_path}'...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The data file '{data_path}' was not found.")
        return
    
    # --- Recreate the exact same test set as during training ---
    # This logic MUST match the v3 training script exactly.
    print("3. Engineering the 'Task_Duration_Hours' target variable...")
    terrain_multipliers = df['Terrain'].map({'Flat': 1.0, 'Incline': 1.15, 'Steep': 1.30})
    df['Task_Duration_Hours'] = ((df['Idling_Time'] / 60) + (df['Load_Cycles'] * 0.1)) * terrain_multipliers
    df = df.dropna()

    # Define the features and target, matching the v3 training script
    target = 'Task_Duration_Hours'
    features = [
        'Machine_ID',
        'Operator_ID',
        'Task_Type',
        'Soil_Type',
        'Terrain',
        'Load_Cycles',
        'Temperature_C',
        'Precipitation_mm',
        'Operator_Experience_Years'
    ]

    X = df[features]
    y = df[target]
    
    # Use the same random_state to get the identical split as in training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("4. Making predictions on the v3 test data...")
    predictions = model_pipeline.predict(X_test)

    # 5. Calculate performance metrics
    print("\n--- Model v3 Performance ---")
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"✅ Mean Absolute Error (MAE): {mae:.4f} hours")
    print(f"✅ Root Mean Squared Error (RMSE): {rmse:.4f} hours")
    print("-------------------------")
    print("(MAE means on average, the prediction is off by this many hours.)")

    # 6. Show a few examples side-by-side
    print("\n--- Sample Predictions vs. Actual Values (v3) ---")
    comparison_df = pd.DataFrame({
        'Actual Duration (Hrs)': y_test, 
        'Predicted Duration (Hrs)': predictions
    })
    comparison_df['Difference (Hrs)'] = comparison_df['Actual Duration (Hrs)'] - comparison_df['Predicted Duration (Hrs)']
    
    print(comparison_df.head(10).to_string())
    print("---------------------------------------------------\n")

if __name__ == "__main__":
    evaluate_time_model_v3()
