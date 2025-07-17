# import pandas as pd
# import numpy as np
# import joblib

# def evaluate_profiler_model(data_path='synthetic_machine_data_v3.csv', model_path='operator_profiler_model.joblib'):
#     """
#     Loads the saved operator profiler model, analyzes the clusters,
#     and dynamically generates personalized feedback for each operator.
#     """
#     print(f"1. Loading profiler model from '{model_path}'...")
#     try:
#         pipeline = joblib.load(model_path)
#     except FileNotFoundError:
#         print(f"Error: The model file '{model_path}' was not found.")
#         print("Please make sure you have trained the profiler model first.")
#         return

#     print(f"2. Loading and preparing data from '{data_path}'...")
#     try:
#         df = pd.read_csv(data_path)
#     except FileNotFoundError:
#         print(f"Error: The data file '{data_path}' was not found.")
#         return

#     # --- Step 1: Recreate Operator Profiles ---
#     print("3. Re-engineering operator performance profiles...")
#     total_hours = df.groupby('Operator_ID')['Engine_Hours'].apply(lambda x: x.max() - x.min())
    
#     operator_profiles = df.groupby('Operator_ID').agg(
#         total_load_cycles=('Load_Cycles', 'sum'),
#         total_fuel_used=('Fuel_Used', 'sum'),
#         total_idling_time_min=('Idling_Time', 'sum'),
#         total_safety_alerts=('Safety_Alert_Triggered', lambda x: (x == 'Yes').sum())
#     ).reset_index()

#     operator_profiles = operator_profiles.merge(total_hours.rename('total_engine_hours'), on='Operator_ID')

#     operator_profiles['fuel_per_load_cycle'] = (operator_profiles['total_fuel_used'] / operator_profiles['total_load_cycles']).replace([np.inf, -np.inf], 0)
#     operator_profiles['idling_ratio'] = (operator_profiles['total_idling_time_min'] / 60 / operator_profiles['total_engine_hours']).replace([np.inf, -np.inf], 0)
#     operator_profiles['safety_incident_rate'] = (operator_profiles['total_safety_alerts'] / operator_profiles['total_engine_hours']).replace([np.inf, -np.inf], 0)
    
#     features_for_clustering = [
#         'fuel_per_load_cycle',
#         'idling_ratio',
#         'safety_incident_rate'
#     ]
#     X = operator_profiles[features_for_clustering].fillna(0)

#     # --- Step 2: Assign Clusters ---
#     print("4. Assigning operators to performance clusters...")
#     operator_profiles['performance_cluster'] = pipeline.predict(X)

#     # --- Step 3: Analyze and Interpret Clusters ---
#     print("\n--- Cluster Analysis ---")
#     cluster_analysis = operator_profiles.groupby('performance_cluster')[features_for_clustering].mean()
#     print("\nAverage Stats per Cluster:")
#     print(cluster_analysis.to_string())

#     # --- Step 4: Dynamically Generate Personalized Feedback ---
#     # This is the new, dynamic section.
#     print("\n--- Dynamically Generated Operator Feedback ---")

#     # Determine which cluster is "best" (lowest scores on all metrics)
#     # We sum the scaled metrics to get a simple ranking score for each cluster
#     scaler = pipeline.named_steps['scaler']
#     scaled_centers = scaler.transform(cluster_analysis)
#     cluster_rank_scores = scaled_centers.sum(axis=1)
    
#     best_cluster_label = cluster_rank_scores.argmin()
#     worst_cluster_label = cluster_rank_scores.argmax()

#     # Define tier names based on dynamic analysis
#     tier_map = {best_cluster_label: "Top Tier", worst_cluster_label: "Needs Coaching"}
    
#     # Get the site-wide averages for comparison
#     global_avg_fuel = operator_profiles['fuel_per_load_cycle'].mean()
#     global_avg_idling = operator_profiles['idling_ratio'].mean()

#     for index, operator in operator_profiles.iterrows():
#         op_id = operator['Operator_ID']
#         cluster = operator['performance_cluster']
        
#         # Assign a tier name, defaulting to "Consistent Performer"
#         tier_name = tier_map.get(cluster, "Consistent Performer")
        
#         print(f"\n--- Profile for {op_id} (Tier: {tier_name}) ---")
        
#         # Dynamic Fuel Insight
#         fuel_diff = ((operator['fuel_per_load_cycle'] / global_avg_fuel) - 1) * 100
#         if fuel_diff < -5: # More than 5% better
#             print(f"  ✅ Fuel Efficiency: Excellent. Your fuel use per load cycle is {abs(fuel_diff):.0f}% lower than the site average.")
#         elif fuel_diff > 5: # More than 5% worse
#             print(f"  ⚠️ Fuel Efficiency: Your fuel use per load cycle is {fuel_diff:.0f}% higher than average. Focus on smoother operations.")
#         else:
#             print("  - Fuel Efficiency: On par with the site average.")

#         # Dynamic Idling Insight
#         idle_diff = ((operator['idling_ratio'] / global_avg_idling) - 1) * 100
#         if idle_diff < -10: # More than 10% better
#             print(f"  ✅ Idling Time: Great job keeping idle time low.")
#         elif idle_diff > 10: # More than 10% worse
#             print(f"  💡 Idling Time: Your idling ratio is {idle_diff:.0f}% higher than average. Consider shutting down during long waits.")
#         else:
#             print("  - Idling Time: On par with the site average.")
            
#         # Dynamic Safety Insight
#         if operator['safety_incident_rate'] == 0:
#             print("  ✅ Safety: Perfect record with zero safety alerts. Keep up the great work!")
#         elif operator['safety_incident_rate'] < operator_profiles['safety_incident_rate'].mean():
#              print("  - Safety: Good record with a low number of safety alerts.")
#         else:
#             print("  ⚠️ Safety: Your rate of safety alerts is higher than average. Please prioritize safety checks.")

# if __name__ == "__main__":
#     evaluate_profiler_model()
import pandas as pd
import numpy as np
import joblib

def generate_insights_for_operator(operator_profile, global_averages, tier_map):
    """
    Generates a structured dictionary of feedback for a single operator.
    This acts as our "API" to get insights on demand.

    Args:
        operator_profile (pd.Series): A single row from the operator_profiles DataFrame.
        global_averages (dict): A dictionary containing site-wide average stats.
        tier_map (dict): A mapping from cluster label to tier name (e.g., {0: "Top Tier"}).

    Returns:
        dict: A dictionary containing the operator's tier and a list of insight strings.
    """
    insights = []
    op_id = operator_profile['Operator_ID']
    cluster = operator_profile['performance_cluster']
    
    # Assign a tier name, defaulting to "Consistent Performer"
    tier_name = tier_map.get(cluster, "Consistent Performer")
    
    # 1. Dynamic Fuel Insight
    fuel_diff = ((operator_profile['fuel_per_load_cycle'] / global_averages['fuel']) - 1) * 100
    if fuel_diff < -5:
        insights.append(f"✅ Fuel Efficiency: Excellent. Your fuel use per load cycle is {abs(fuel_diff):.0f}% lower than the site average.")
    elif fuel_diff > 5:
        insights.append(f"⚠️ Fuel Efficiency: Your fuel use per load cycle is {fuel_diff:.0f}% higher than average. Focus on smoother operations.")
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
        insights.append("⚠️ Safety: Your rate of safety alerts is higher than average. Please prioritize safety checks.")
        
    return {
        "operator_id": op_id,
        "tier": tier_name,
        "insights": insights
    }


def evaluate_profiler_model(data_path='synthetic_machine_data_v3.csv', model_path='operator_profiler_model.joblib'):
    """
    Loads the saved operator profiler model, analyzes the clusters,
    and uses a dedicated function to generate personalized feedback.
    """
    print(f"1. Loading profiler model from '{model_path}'...")
    try:
        pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: The model file '{model_path}' was not found.")
        return

    print(f"2. Loading and preparing data from '{data_path}'...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The data file '{data_path}' was not found.")
        return

    # --- Step 1: Recreate Operator Profiles ---
    print("3. Re-engineering operator performance profiles...")
    total_hours = df.groupby('Operator_ID')['Engine_Hours'].apply(lambda x: x.max() - x.min())
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
    print("4. Assigning operators to performance clusters...")
    operator_profiles['performance_cluster'] = pipeline.predict(X)
    cluster_analysis = operator_profiles.groupby('performance_cluster')[features_for_clustering].mean()
    print("\nAverage Stats per Cluster:")
    print(cluster_analysis.to_string())

    # --- Step 3: Prepare for Insight Generation ---
    scaler = pipeline.named_steps['scaler']
    scaled_centers = scaler.transform(cluster_analysis)
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
    all_operator_feedback = {}
    for index, operator in operator_profiles.iterrows():
        # Use our new "API" function to get the feedback
        feedback = generate_insights_for_operator(operator, global_averages, tier_map)
        all_operator_feedback[operator['Operator_ID']] = feedback
        
        # Display the generated feedback
        print(f"\n--- Profile for {feedback['operator_id']} (Tier: {feedback['tier']}) ---")
        for insight in feedback['insights']:
            print(f"  {insight}")

    # --- Step 5: Demonstrate API-like Usage ---
    print("\n\n--- API-like Usage Example ---")
    print("To get insights for a single operator (e.g., OP1001), you can now access the stored dictionary:")
    op1001_feedback = all_operator_feedback.get('OP1001', {})
    if op1001_feedback:
        print(f"Insights for OP1001: {op1001_feedback['insights']}")


if __name__ == "__main__":
    evaluate_profiler_model()
