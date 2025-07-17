import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- Initial State for Realistic Data ---
MACHINE_IDS = [f"EXC00{i}" for i in range(1, 6)]
OPERATOR_IDS = [f"OP100{i}" for i in range(1, 6)]
TERRAIN_TYPES = ['Flat', 'Incline', 'Steep']

# V4: Machine states track engine hours and current temperature
MACHINE_STATES = {
    machine_id: {
        'engine_hours': np.random.uniform(1000, 3000),
        'temperature_c': 85.0 # Starting temp
    } for machine_id in MACHINE_IDS
}

current_timestamp = datetime(2025, 5, 1, 8, 0, 0)
# ---

# 1. V4: Define 5 scenarios covering the expanded Task and Soil types
scenarios = [
    {
        "name": "Digging Rock", "weight": 0.20,
        "rules": {
            "Task_Type": "Digging", "Soil_Type": "Rock", "Precipitation_mm": 0,
            "Load_Cycles": lambda: np.random.randint(15, 25), "Base_Fuel_Per_Cycle": 0.5, "Base_Idle_Time": 15
        }
    },
    {
        "name": "Hauling Sand", "weight": 0.20,
        "rules": {
            "Task_Type": "Hauling", "Soil_Type": "Sand", "Precipitation_mm": 0,
            "Load_Cycles": lambda: np.random.randint(5, 15), "Base_Fuel_Per_Cycle": 0.4, "Base_Idle_Time": 30
        }
    },
    {
        "name": "Trenching Loam", "weight": 0.20, # NEW SCENARIO
        "rules": {
            "Task_Type": "Trenching", "Soil_Type": "Loam", "Precipitation_mm": 0,
            "Load_Cycles": lambda: np.random.randint(8, 18), "Base_Fuel_Per_Cycle": 0.45, "Base_Idle_Time": 25
        }
    },
    {
        "name": "Compacting Gravel", "weight": 0.20, # NEW SCENARIO
        "rules": {
            "Task_Type": "Compacting", "Soil_Type": "Gravel", "Precipitation_mm": 0,
            "Load_Cycles": lambda: np.random.randint(1, 5), "Base_Fuel_Per_Cycle": 0.8, "Base_Idle_Time": 20 # High fuel use for constant movement
        }
    },
    {
        "name": "Grading Clay (Rainy)", "weight": 0.20,
        "rules": {
            "Task_Type": "Grading", "Soil_Type": "Clay", "Precipitation_mm": lambda: np.random.uniform(10, 50), # Re-added Precipitation
            "Load_Cycles": lambda: np.random.randint(1, 5), "Base_Fuel_Per_Cycle": 0.3, "Base_Idle_Time": 60
        }
    }
]

# 2. Generate the data
num_rows = 2000
data = []
scenario_list = [s['name'] for s in scenarios]
scenario_weights = [s['weight'] for s in scenarios]

for _ in range(num_rows):
    # --- Assign base IDs and choose a scenario ---
    machine_id = random.choice(MACHINE_IDS)
    operator_id = random.choice(OPERATOR_IDS)
    terrain = random.choice(TERRAIN_TYPES)
    
    chosen_scenario_name = random.choices(scenario_list, weights=scenario_weights, k=1)[0]
    scenario = next(s for s in scenarios if s['name'] == chosen_scenario_name)
    rules = scenario['rules']
    
    row = {}
    
    # --- Calculate metrics with terrain multipliers ---
    terrain_multiplier = 1.0 + (0.15 if terrain == 'Incline' else 0.30 if terrain == 'Steep' else 0.0)

    load_cycles = rules['Load_Cycles']()
    idling_time_min = rules['Base_Idle_Time'] * np.random.uniform(0.9, 1.1)
    
    fuel_used = (5 + (load_cycles * rules['Base_Fuel_Per_Cycle'])) * terrain_multiplier
    task_duration_hours = ((idling_time_min / 60) + (load_cycles * 0.1)) * terrain_multiplier
    
    # --- Stateful Temperature Logic ---
    current_temp = MACHINE_STATES[machine_id]['temperature_c']
    temp_increase = task_duration_hours * np.random.uniform(4, 6)
    current_temp += temp_increase
    
    # --- Fill in the row data ---
    row['Timestamp'] = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    row['Machine_ID'] = machine_id
    row['Operator_ID'] = operator_id
    row['Engine_Hours'] = round(MACHINE_STATES[machine_id]['engine_hours'], 2)
    row['Fuel_Used'] = round(fuel_used, 2)
    row['Load_Cycles'] = load_cycles
    row['Idling_Time'] = round(idling_time_min)
    row['Task_Type'] = rules['Task_Type']
    row['Soil_Type'] = rules['Soil_Type']
    row['Terrain'] = terrain
    row['Temperature_C'] = round(current_temp, 2)
    row['Precipitation_mm'] = round(rules['Precipitation_mm']() if callable(rules['Precipitation_mm']) else rules['Precipitation_mm'], 2)
    row['Operator_Experience_Years'] = np.random.randint(1, 20) # Simplified for v4
    
    # --- Update state for the NEXT iteration ---
    MACHINE_STATES[machine_id]['engine_hours'] += task_duration_hours
    break_duration_hours = np.random.uniform(0.25, 1.0)
    MACHINE_STATES[machine_id]['temperature_c'] = max(85.0, current_temp - (break_duration_hours * 2)) # Cools down, but not below 85
    current_timestamp += timedelta(hours=(task_duration_hours + break_duration_hours))
    
    # --- Generate safety data ---
    row['Seatbelt_Status'] = 'Unfastened' if row['Operator_Experience_Years'] < 3 and random.random() < 0.2 else 'Fastened'
    row['Safety_Alert_Triggered'] = 'Yes' if row['Seatbelt_Status'] == 'Unfastened' else 'No'

    data.append(row)

# 3. Create the final DataFrame
final_columns = [
    'Timestamp', 'Machine_ID', 'Operator_ID', 'Engine_Hours', 
    'Fuel_Used', 'Load_Cycles', 'Idling_Time', 'Seatbelt_Status', 
    'Safety_Alert_Triggered', 'Task_Type', 'Soil_Type', 'Terrain',
    'Temperature_C', 'Precipitation_mm', 'Operator_Experience_Years'
]
df = pd.DataFrame(data)[final_columns]

# Save the new, comprehensive dataset
df.to_csv('synthetic_machine_data_v3.csv', index=False)

print("Successfully generated 'synthetic_machine_data_v4.csv' with 5x Task Types, 5x Soil Types, Terrain, and Stateful Temperature.")
print("\nSample of the new data:")
print(df.head().to_string())
