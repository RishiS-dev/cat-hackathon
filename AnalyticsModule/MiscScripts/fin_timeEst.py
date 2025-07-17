import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def train_final_time_model(data_path='fin_synthetic_machine_data.csv'):
    """
    Loads the final dataset and trains the definitive model to predict
    task duration, now including the 'RPM' feature.
    """
    print(f"1. Loading final data from '{data_path}'...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        print("Please make sure you have generated the final dataset first.")
        return

    # --- Feature Engineering ---
    # This calculation MUST exactly match the logic from your final data generation script.
    print("2. Engineering the 'Task_Duration_Hours' target variable...")
    terrain_multipliers = df['Terrain'].map({'Flat': 1.0, 'Incline': 1.15, 'Steep': 1.30})
    df['Task_Duration_Hours'] = ((df['Idling_Time'] / 60) + (df['Load_Cycles'] * 0.1)) * terrain_multipliers
    df = df.dropna()

    # 3. Define features (X) and target (y)
    target = 'Task_Duration_Hours'

    # The final feature set now includes 'RPM' for more accuracy
    features = [
        'Machine_ID',
        'Operator_ID',
        'RPM',  # NEW: Added the crucial RPM feature
        'Task_Type',
        'Soil_Type',
        'Terrain',
        'Load_Cycles',
        'Temperature_C',
        'Precipitation_mm'
        # 'Operator_Experience_Years' is removed as its effect is now captured by Operator_ID personas
    ]
    
    X = df[features]
    y = df[target]
    
    print("3. Preprocessing data (including new 'RPM' feature)...")
    # Define which features are categorical and need encoding
    categorical_features = ['Machine_ID', 'Operator_ID', 'Task_Type', 'Soil_Type', 'Terrain']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep numerical columns (like RPM) as they are
    )

    # 4. Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Create and train the model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    print("4. Training the Final RandomForest model for Task Duration...")
    model_pipeline.fit(X_train, y_train)

    # 6. Evaluate the model
    print("5. Evaluating model performance...")
    predictions = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"-> Final Model RMSE (in hours): {rmse:.4f}")

    # 7. Save the final, definitive model
    model_filename = 'fin_task_duration_model.joblib'
    joblib.dump(model_pipeline, model_filename)
    print(f"6. Final time estimation model saved successfully as '{model_filename}'")


if __name__ == "__main__":
    train_final_time_model()
