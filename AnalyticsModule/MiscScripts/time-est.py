import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

def train_time_model_v3(data_path='synthetic_machine_data_v3.csv'):
    """
    Loads the v3 dataset and trains an updated model to predict
    task duration, now including the 'Terrain' feature.
    """
    print(f"1. Loading data from '{data_path}'...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        print("Please make sure you have generated the v3 dataset first.")
        return

    # --- Feature Engineering ---
    # This calculation MUST match the logic from your datagen_v3.py script
    # to create the correct target variable.
    print("2. Engineering the 'Task_Duration_Hours' target variable...")
    terrain_multipliers = df['Terrain'].map({'Flat': 1.0, 'Incline': 1.15, 'Steep': 1.30})
    df['Task_Duration_Hours'] = ((df['Idling_Time'] / 60) + (df['Load_Cycles'] * 0.1)) * terrain_multipliers
    df = df.dropna()

    # 3. Define features (X) and target (y)
    target = 'Task_Duration_Hours'

    # The feature set now includes 'Terrain' for more accurate predictions
    features = [
        'Machine_ID',
        'Operator_ID',
        'Task_Type',
        'Soil_Type',
        'Terrain',  # NEW: Added the crucial Terrain feature
        'Load_Cycles',
        'Temperature_C',
        'Precipitation_mm',
        'Operator_Experience_Years'
    ]
    
    X = df[features]
    y = df[target]
    
    print("3. Preprocessing data (including new 'Terrain' feature)...")
    # Add 'Terrain' to the list of features that need to be one-hot encoded
    categorical_features = ['Machine_ID', 'Operator_ID', 'Task_Type', 'Soil_Type', 'Terrain']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep the numerical columns as they are
    )

    # 4. Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Create and train the model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    print("4. Training the v3 RandomForest model for Task Duration...")
    model_pipeline.fit(X_train, y_train)

    # 6. Evaluate the model
    print("5. Evaluating model performance...")
    predictions = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"-> Model v3 RMSE (in hours): {rmse:.4f}")

    # 7. Save the new, more powerful model
    model_filename = 'task_duration_model_v3.joblib'
    joblib.dump(model_pipeline, model_filename)
    print(f"6. Model v3 saved successfully as '{model_filename}'")


if __name__ == "__main__":
    train_time_model_v3()
