import pandas as pd
import numpy as np
import joblib
import os

def generate_insights_for_operator(operator_profile, global_averages, tier_map):
    """
    Generates a structured dictionary of feedback for a single operator.
    This acts as our "API" to get insights on demand.
    """
    insights = []
    op_id = operator_profile['Operator_ID']
    cluster = operator_profile['performance_cluster']
    
    tier_name = tier_map.get(cluster, "Consistent Performer")
    
    # 1. Dynamic Fuel Insight
    fuel_diff = ((operator_profile['fuel_per_load_cycle'] / global_averages['fuel']) - 1) * 100
    if fuel_diff < -5:
        insights.append(f"✅ Fuel Efficiency: Excellent. Your fuel use is {abs(fuel_diff):.0f}% lower than the site average.")
    elif fuel_diff > 5:
        insights.append(f"⚠️ Fuel Efficiency: Your fuel use is {fuel_diff:.0f}% higher than average. Focus on smoother operations.")
    else:
        insights.append("- Fuel Efficiency: On par with the site average.")

    # 2. Dynamic Idling Insight
    idle_diff = ((operator_profile['idling_ratio'] / global_averages['idling']) - 1) * 100
    if idle_diff < -10:
        insights.append("✅ Idling Time: Great job keeping idle time low.")
    elif idle_diff > 10:
        insights.append(f"💡 Idling Time: Your idling ratio is {idle_diff:.0f}% higher than average. Consider shutting down during long waits.")
    else:
        insights.append("- Idling Time: On par with the site average.")
        
    # 3. Dynamic Safety Insight
    if operator_profile['safety_incident_rate'] == 0:
        insights.append("✅ Safety: Perfect record with zero safety alerts. Keep up the great work!")
    elif operator_profile['safety_incident_rate'] < global_averages['safety']:
         insights.append("- Safety: Good record with a low number of safety alerts.")
    else:
        insights.append(f"⚠️ Safety: Your rate of safety alerts is {((operator_profile['safety_incident_rate']/global_averages['safety'])-1)*100:.0f}% higher than average. Please prioritize safety checks.")
        
    return {
        "operator_id": op_id,
        "tier": tier_name,
        "insights": insights
    }


def evaluate_final_profiler_model(data_path='fin_synthetic_machine_data.csv', model_path='fin_operator_profiler_model.joblib'):
    """
    Loads the final profiler model, analyzes the clusters,
    and uses a dedicated function to generate personalized feedback.
    """
    print(f"1. Loading final profiler model from '{model_path}'...")
    try:
        BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        pipeline = joblib.load(os.path.join(BASE_DIR, model_path))
        df = pd.read_csv(os.path.join(BASE_DIR, data_path))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Make sure the required files are in the same directory as the script.")
        return

    # --- Step 1: Recreate Operator Profiles ---
    print("2. Re-engineering operator performance profiles...")
    total_hours = df.groupby('Operator_ID')['Engine_Hours'].apply(lambda x: x.max() - x.min()).replace(0, 1)
    operator_profiles = df.groupby('Operator_ID').agg(
        total_load_cycles=('Load_Cycles', 'sum'),
        total_fuel_used=('Fuel_Used', 'sum'),
        total_idling_time_min=('Idling_Time', 'sum'),
        total_safety_alerts=('Safety_Alert_Triggered', lambda x: (x == 'Yes').sum())
    ).reset_index()
    operator_profiles = operator_profiles.merge(total_hours.rename('total_engine_hours'), on='Operator_ID')
    operator_profiles['fuel_per_load_cycle'] = (operator_profiles['total_fuel_used'] / operator_profiles['total_load_cycles']).replace([np.inf, -np.inf], 0)
    operator_profiles['idling_ratio'] = (operator_profiles['total_idling_time_min'] / 60 / operator_profiles['total_engine_hours']).replace([np.inf, -np.inf], 0)
    operator_profiles['safety_incident_rate'] = (operator_profiles['total_safety_alerts'] / operator_profiles['total_engine_hours']).replace([np.inf, -np.inf], 0)
    features_for_clustering = ['fuel_per_load_cycle', 'idling_ratio', 'safety_incident_rate']
    X = operator_profiles[features_for_clustering].fillna(0)

    # --- Step 2: Assign Clusters and Analyze ---
    print("3. Assigning operators to performance clusters...")
    operator_profiles['performance_cluster'] = pipeline.predict(X)
    cluster_analysis = operator_profiles.groupby('performance_cluster')[features_for_clustering].mean()
    print("\n--- Cluster Analysis (Average Stats per Cluster) ---")
    print(cluster_analysis.to_string())

    # --- Step 3: Prepare for Insight Generation ---
    scaler = pipeline.named_steps['scaler']
    scaled_centers = scaler.transform(cluster_analysis)
    
    # We sum the scaled metrics to get a simple ranking score for each cluster.
    # Lower is better.
    cluster_rank_scores = scaled_centers.sum(axis=1)
    
    best_cluster_label = cluster_rank_scores.argmin()
    worst_cluster_label = cluster_rank_scores.argmax()
    
    tier_map = {best_cluster_label: "Top Tier", worst_cluster_label: "Needs Coaching"}
    
    global_averages = {
        'fuel': operator_profiles['fuel_per_load_cycle'].mean(),
        'idling': operator_profiles['idling_ratio'].mean(),
        'safety': operator_profiles['safety_incident_rate'].mean()
    }

    # --- Step 4: Generate and Display Feedback for All Operators ---
    print("\n--- Dynamically Generated Operator Feedback ---")
    for index, operator in operator_profiles.iterrows():
        feedback = generate_insights_for_operator(operator, global_averages, tier_map)
        
        print(f"\n--- Profile for {feedback['operator_id']} (Tier: {feedback['tier']}) ---")
        for insight in feedback['insights']:
            print(f"  {insight}")

if __name__ == "__main__":
    evaluate_final_profiler_model()
