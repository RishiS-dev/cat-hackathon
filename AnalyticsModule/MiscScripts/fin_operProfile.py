import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import joblib
import os

def train_final_profiler_model(data_path='fin_synthetic_machine_data.csv'):
    """
    Loads the final machine data, engineers performance features for each 
    operator, and uses KMeans clustering to group them into performance tiers.
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

    # --- Step 1: Feature Engineering ---
    # Create a "profile" for each operator by calculating their average performance.
    print("2. Engineering operator performance profiles...")

    # Calculate total engine hours for each operator to normalize other metrics
    total_hours = df.groupby('Operator_ID')['Engine_Hours'].apply(lambda x: x.max() - x.min()).replace(0, 1)
    
    operator_profiles = df.groupby('Operator_ID').agg(
        total_load_cycles=('Load_Cycles', 'sum'),
        total_fuel_used=('Fuel_Used', 'sum'),
        total_idling_time_min=('Idling_Time', 'sum'),
        total_safety_alerts=('Safety_Alert_Triggered', lambda x: (x == 'Yes').sum())
    ).reset_index()

    # Join total hours back to profiles
    operator_profiles = operator_profiles.merge(total_hours.rename('total_engine_hours'), on='Operator_ID')

    # Create meaningful, comparable metrics (the "DNA" of the operator)
    operator_profiles['fuel_per_load_cycle'] = (operator_profiles['total_fuel_used'] / operator_profiles['total_load_cycles']).replace([np.inf, -np.inf], 0)
    operator_profiles['idling_ratio'] = (operator_profiles['total_idling_time_min'] / 60 / operator_profiles['total_engine_hours']).replace([np.inf, -np.inf], 0)
    operator_profiles['safety_incident_rate'] = (operator_profiles['total_safety_alerts'] / operator_profiles['total_engine_hours']).replace([np.inf, -np.inf], 0)
    
    # Select only the engineered features for clustering
    features_for_clustering = [
        'fuel_per_load_cycle',
        'idling_ratio',
        'safety_incident_rate'
    ]
    X = operator_profiles[features_for_clustering].fillna(0)

    print("   - Operator profiles created successfully.")
    print(X.head())

    # --- Step 2 & 3: Scaling and Clustering with a Pipeline ---
    # Create a pipeline to first scale the data, then apply KMeans.
    print("\n3. Building the K-Means clustering pipeline...")

    # We will find 3 clusters based on our persona design
    N_CLUSTERS = 3

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10))
    ])

    print(f"4. Fitting the model to find {N_CLUSTERS} operator clusters...")
    pipeline.fit(X)

    # --- Step 4: Assign Clusters and Save Model ---
    operator_profiles['performance_cluster'] = pipeline.predict(X)
    
    model_filename = 'fin_operator_profiler_model.joblib'
    joblib.dump(pipeline, os.path.join(BASE_DIR, model_filename))
    print(f"\n5. Final profiler model saved successfully as '{model_filename}'")

    # --- Step 5: Display Results ---
    print("\n--- Operator Profiling Results ---")
    # Sort by cluster to see the groups clearly
    print(operator_profiles[['Operator_ID', 'performance_cluster']].sort_values('performance_cluster').to_string(index=False))
    print("\nNext step: Evaluate this model to interpret what each cluster means.")


if __name__ == "__main__":
    train_final_profiler_model()
