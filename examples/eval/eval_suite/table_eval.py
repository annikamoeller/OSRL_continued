import os
import pandas as pd
import numpy as np
import sys
import argparse # <-- ADDED

# Ensure the project root is in the path
PROJECT_ROOT = "/home/20234949/thesis/OSRL_continued"
sys.path.insert(0, PROJECT_ROOT)

# --- GLOBAL MASTER CONFIGURATION ---
# We keep this static because it's your master reference file, not run-specific
STATS_CSV = os.path.join(PROJECT_ROOT, "dataset_analysis", "master_dataset_stats.csv")
EPSILON = 1e-8

def generate_table(run_dir):
    # Dynamically build paths based on the run folder
    raw_data_csv = os.path.join(run_dir, "raw_data.csv")
    output_table_csv = os.path.join(run_dir, "publication_table_kappa10.csv")

    if not os.path.exists(raw_data_csv):
        raise FileNotFoundError(f"❌ Missing {raw_data_csv}. Did the collection script fail?")
    if not os.path.exists(STATS_CSV):
        raise FileNotFoundError(f"❌ Missing {STATS_CSV}. Please check your dataset analysis folder.")
        
    raw_df = pd.read_csv(raw_data_csv)
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
        # Note: If target cost is 0, we add epsilon to avoid dividing by zero
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
    
    # Save to the specific run folder
    summary.to_csv(output_table_csv)
    
    # Print a clean terminal output
    print("\n" + "═"*70)
    print("📊 PUBLICATION SUMMARY TABLE (Target Cost κ = 10.0)")
    print("═"*70)
    print(summary.to_string())
    print("═"*70)
    print(f"✅ Table saved to: {output_table_csv}")

if __name__ == "__main__":
    # Setup Argument Parser to accept the FOLDER path
    parser = argparse.ArgumentParser(description="Generate Publication Tables from Eval Data")
    parser.add_argument("run_dir", type=str, help="Path to the timestamped run folder (e.g., examples/eval/eval_suite/eval_20260326_1436)")
    args = parser.parse_args()

    run_dir = args.run_dir
    
    # Check if the folder exists
    if not os.path.isdir(run_dir):
        print(f"❌ Error: The directory '{run_dir}' does not exist.")
        sys.exit(1)

    # Execute the table generation
    generate_table(run_dir)