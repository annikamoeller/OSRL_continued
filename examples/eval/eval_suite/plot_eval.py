import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import argparse

# Ensure the project root is in the path
PROJECT_ROOT = "/home/20234949/thesis/OSRL_continued"
sys.path.insert(0, PROJECT_ROOT)

# Ensure the project root is in the path
PROJECT_ROOT = "/home/20234949/thesis/OSRL_continued"
sys.path.insert(0, PROJECT_ROOT)

STATS_CSV = os.path.join(PROJECT_ROOT, "dataset_analysis", "master_dataset_stats.csv")
EPSILON = 1e-8

def load_data(raw_data_csv):
    if not os.path.exists(raw_data_csv):
        raise FileNotFoundError(f"Missing {raw_data_csv}. Run data collection first.")
    if not os.path.exists(STATS_CSV):
        raise FileNotFoundError(f"Missing {STATS_CSV}. Run dataset analysis first.")
    
    raw_df = pd.read_csv(raw_data_csv)
    stats_df = pd.read_csv(STATS_CSV)
    
    # Convert stats to a dictionary for easy lookup
    stats_lookup = stats_df.set_index("Task").to_dict('index')
    return raw_df, stats_lookup

def process_and_normalize(raw_df, stats_lookup):
    """Calculates Normalized Reward using the min/max from your dataset stats."""
    processed_records = []
    
    for _, row in raw_df.iterrows():
        # Clean the task name to match the stats CSV (e.g., "AntRun")
        task_name = row["Task"].replace("Offline", "").replace("-v0", "")
        
        # Fuzzy match to find the corresponding stats
        match = next((k for k in stats_lookup.keys() if task_name in k), None)
        
        if match:
            r_max = stats_lookup[match]["Return_Max"]
            r_min = stats_lookup[match]["Return_Min"]
            median_cost = stats_lookup[match]["Cost_Median"]
        else:
            print(f"⚠️ No stats match for {task_name}. Using defaults.")
            r_max, r_min, median_cost = 1000.0, 0.0, 10.0
            
        # Standard Paper Normalization
        norm_reward = ((row["Raw_Eval_Reward"] - r_min) / (r_max - r_min + EPSILON)) * 100
        
        record = row.to_dict()
        record["Clean_Task"] = task_name
        record["Norm_Reward"] = norm_reward
        record["Dataset_Median_Cost"] = median_cost
        processed_records.append(record)
        
    return pd.DataFrame(processed_records)

def create_thesis_plot(df, output_plot_path):
    """Generates the grid of plots (Row 1: Reward, Row 2: Cost)."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    tasks = sorted(df['Clean_Task'].unique())
    num_tasks = len(tasks)
    
    # Create a 2 x N grid
    fig, axes = plt.subplots(2, num_tasks, figsize=(4.5 * num_tasks, 8), sharex=True)
    if num_tasks == 1: 
        axes = axes.reshape(2, 1)

    for i, task in enumerate(tasks):
        task_df = df[df['Clean_Task'] == task]
        median_ds_cost = task_df['Dataset_Median_Cost'].iloc[0]
        
        # --- ROW 1: Normalized Reward ---
        sns.lineplot(
            ax=axes[0, i], data=task_df, 
            x="Target_Cost", y="Norm_Reward", 
            hue="Variant", marker="o", 
            legend=(i == num_tasks - 1) # Only put legend on the last plot
        )
        axes[0, i].set_title(task, fontweight='bold')
        axes[0, i].set_ylabel("Normalized Reward (%)" if i == 0 else "")
        
        # Vertical line for median dataset cost
        axes[0, i].axvline(x=median_ds_cost, color='red', linestyle='--', alpha=0.5, label="Dataset Median Cost")
        
        # --- ROW 2: Evaluated Cost ---
        sns.lineplot(
            ax=axes[1, i], data=task_df, 
            x="Target_Cost", y="Raw_Eval_Cost", 
            hue="Variant", marker="s", legend=False
        )
        # Ideal Y=X constraint line
        sweep_vals = sorted(task_df["Target_Cost"].unique())
        axes[1, i].plot(sweep_vals, sweep_vals, 'k:', alpha=0.6, label="Ideal (Target = Actual)")
        
        axes[1, i].set_ylabel("Actual Evaluated Cost" if i == 0 else "")
        axes[1, i].set_xlabel("Target Cost Prompt")
        
        # Vertical line for median dataset cost
        axes[1, i].axvline(x=median_ds_cost, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot successfully generated: {output_plot_path}")

if __name__ == "__main__":
    import argparse
    
    # Setup Argument Parser to accept the FOLDER path
    parser = argparse.ArgumentParser(description="Plot Safe RL Eval Results")
    parser.add_argument("run_dir", type=str, help="Path to the timestamped run folder")
    args = parser.parse_args()

    run_dir = args.run_dir
    
    # Check if the folder exists
    if not os.path.isdir(run_dir):
        print(f"❌ Error: The directory '{run_dir}' does not exist.")
        sys.exit(1)

    # Standardized paths inside the specific run folder
    raw_csv_path = os.path.join(run_dir, "raw_data.csv")
    processed_csv_path = os.path.join(run_dir, "processed_data.csv")
    output_plot_path = os.path.join(run_dir, "eval_plot.png")

    # Execute Pipeline
    raw_df, stats_lookup = load_data(raw_csv_path)
    processed_df = process_and_normalize(raw_df, stats_lookup)
    
    # Save processed data and plot inside the same folder
    processed_df.to_csv(processed_csv_path, index=False)
    create_thesis_plot(processed_df, output_plot_path)