import os
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_CSV = os.path.join(SCRIPT_DIR, "raw_eval_collection.csv")
# Note: Ensure this path correctly points to your absolute dataset analysis folder
STATS_CSV = "/home/20234949/thesis/OSRL_continued/dataset_analysis/master_dataset_stats.csv"
OUTPUT_TABLE_CSV = os.path.join(SCRIPT_DIR, "publication_table_kappa10.csv")

EPSILON = 1e-8

def generate_table():
    if not os.path.exists(RAW_DATA_CSV):
        raise FileNotFoundError(f"Missing {RAW_DATA_CSV}. Run data collection first.")
    if not os.path.exists(STATS_CSV):
        raise FileNotFoundError(f"Missing {STATS_CSV}. Please check the absolute path.")
        
    raw_df = pd.read_csv(RAW_DATA_CSV)
    stats_df = pd.read_csv(STATS_CSV)
    stats_lookup = stats_df.set_index("Task").to_dict('index')

    # 1. Filter for the standard baseline prompt (Target Cost = 10)
    # If you want a different target, simply change this value
    table_data = raw_df[raw_df['Target_Cost'] == 10.0].copy()
    
    if table_data.empty:
        print("⚠️ No data found for Target Cost = 10.0.")
        return

    processed_records = []
    
    # 2. Apply Normalization Math
    for _, row in table_data.iterrows():
        task_name = row["Task"].replace("Offline", "").replace("-v0", "")
        match = next((k for k in stats_lookup.keys() if task_name in k), None)
        
        if match:
            r_max = stats_lookup[match]["Return_Max"]
            r_min = stats_lookup[match]["Return_Min"]
        else:
            r_max, r_min = 1000.0, 0.0
            
        # Paper Formula for Reward: ((R - min) / (max - min)) * 100
        norm_reward = ((row["Raw_Eval_Reward"] - r_min) / (r_max - r_min + EPSILON)) * 100
        
        # Paper Formula for Table Cost: Actual Cost / Target Kappa
        norm_cost = row["Raw_Eval_Cost"] / (row["Target_Cost"] + EPSILON)
        
        processed_records.append({
            "Task": task_name,
            "Variant": row["Variant"],
            "Norm_Reward": norm_reward,
            "Norm_Cost": norm_cost
        })

    eval_df = pd.DataFrame(processed_records)

    # 3. Group by Task and Variant, calculate Mean and Std Dev
    summary = eval_df.groupby(['Task', 'Variant']).agg({
        'Norm_Reward': ['mean', 'std'],
        'Norm_Cost': ['mean', 'std']
    }).round(2)
    
    # Flatten multi-level columns (e.g., 'Norm_Reward', 'mean' -> 'Norm_Reward_mean')
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    
    # Save to CSV
    summary.to_csv(OUTPUT_TABLE_CSV)
    
    # Print a clean terminal output
    print("\n" + "═"*70)
    print("📊 PUBLICATION SUMMARY TABLE (Target Cost κ = 10.0)")
    print("═"*70)
    print(summary.to_string())
    print("═"*70)
    print(f"✅ Table saved to: {OUTPUT_TABLE_CSV}")

if __name__ == "__main__":
    generate_table()