import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import h5py
import pandas as pd
import glob

def analyze_dataset(file_path, task_name, output_dir):
    """Processes a single dataset, saves a plot, and returns stats."""
    
    # Clean the task name for plotting (removes -v0, -v1, etc.)
    clean_task_name = task_name.split('-v')[0]
    print(f"🔎 Scanning: {clean_task_name}...")
    
    try:
        # 1. Load Data
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                dataset = pickle.load(f)
        elif file_path.endswith(('.hdf5', '.h5')):
            dataset = {}
            with h5py.File(file_path, 'r') as f:
                for key in f.keys():
                    dataset[key] = f[key][()]
        else:
            return None

        # 2. Slice Trajectories
        terminals_key = "terminals" if "terminals" in dataset else "dones"
        dones = np.logical_or(dataset[terminals_key], dataset.get("timeouts", np.zeros_like(dataset[terminals_key])))
        done_idx = np.where(dones)[0]
        
        trajectories = []
        start_idx = 0
        for end_idx in done_idx:
            end_idx += 1 
            trajectories.append({
                "rewards": dataset["rewards"][start_idx:end_idx],
                "costs": dataset["costs"][start_idx:end_idx],
                "length": end_idx - start_idx
            })
            start_idx = end_idx

        returns = np.array([np.sum(t["rewards"]) for t in trajectories])
        costs = np.array([np.sum(t["costs"]) for t in trajectories])
        lengths = np.array([t["length"] for t in trajectories])
        
        # 3. Plotting
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Cleaned up Title
        total_transitions = sum(lengths)
        fig.suptitle(f"Dataset Analysis: {clean_task_name} | N = {len(trajectories)} Trajectories | Transitions = {total_transitions}", fontsize=16, weight='bold')

        # --- Subplot 1: Histograms ---
        sns.histplot(returns, color="blue", alpha=0.4, label="Returns", kde=True, ax=axes[0])
        sns.histplot(costs, color="red", alpha=0.4, label="Costs", kde=True, ax=axes[0])
        axes[0].set_title("Distribution of Episode Returns and Costs")
        axes[0].legend()

        # --- Subplot 2: Return vs Cost (With Pareto Frontier) ---
        sns.kdeplot(x=costs, y=returns, color="gray", alpha=0.4, levels=8, ax=axes[1]) # Added KDE for density context
        sns.scatterplot(x=costs, y=returns, color="purple", alpha=0.6, label="Trajectories", ax=axes[1])
        
        # Calculate Empirical Reward Frontier (Binned upper envelope)
        num_bins = 20
        bins = np.linspace(costs.min(), costs.max(), num_bins + 1)
        bin_centers = []
        frontier_returns = []
        
        for i in range(num_bins):
            mask = (costs >= bins[i]) & (costs <= bins[i+1])
            if np.any(mask):
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                # Take the 98th percentile to avoid massive outliers ruining the envelope
                frontier_returns.append(np.percentile(returns[mask], 98)) 
                
        # Make the frontier monotonically increasing
        monotonic_frontier = np.maximum.accumulate(frontier_returns)
        axes[1].plot(bin_centers, monotonic_frontier, color="darkorange", marker="o", linewidth=3, label="Empirical Reward Frontier")
        
        axes[1].set_title("Pareto Distribution (Return vs. Cost)")
        axes[1].set_xlabel("Total Episode Cost")
        axes[1].set_ylabel("Total Episode Return")
        axes[1].legend()

        # --- Subplot 3: Accumulation over Time (With Standard Deviation) ---
        max_len = np.max(lengths)
        cum_r = np.full((len(trajectories), max_len), np.nan)
        cum_c = np.full((len(trajectories), max_len), np.nan)
        
        for i, t in enumerate(trajectories):
            cum_r[i, :t["length"]] = np.cumsum(t["rewards"])
            cum_c[i, :t["length"]] = np.cumsum(t["costs"])
        
        # Calculate Means and Standard Deviations ignoring NaNs (uneven sequence lengths)
        mean_r = np.nanmean(cum_r, axis=0)
        std_r = np.nanstd(cum_r, axis=0)
        mean_c = np.nanmean(cum_c, axis=0)
        std_c = np.nanstd(cum_c, axis=0)
        
        t_steps = np.arange(max_len)
        
        # Plot Means
        axes[2].plot(t_steps, mean_r, color="blue", linewidth=2, label="Avg Cumulative Reward")
        axes[2].plot(t_steps, mean_c, color="red", linewidth=2, label="Avg Cumulative Cost")
        
        # Fill Standard Deviation
        axes[2].fill_between(t_steps, mean_r - std_r, mean_r + std_r, color="blue", alpha=0.15)
        axes[2].fill_between(t_steps, mean_c - std_c, mean_c + std_c, color="red", alpha=0.15)
        
        axes[2].set_title("Average Accumulation over Time")
        axes[2].set_xlabel("Timestep")
        axes[2].set_ylabel("Cumulative Value")
        axes[2].legend(loc="upper left")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"dataset_analysis_{clean_task_name}.png"), dpi=300)
        plt.close()

        # 4. Return Data
        return {
            "Task": task_name,
            "Trajectories": len(trajectories),
            "Avg_Length": int(np.mean(lengths)),
            "Return_Max": round(np.max(returns), 2),
            "Return_Min": round(np.min(returns), 2),
            "Cost_Max": round(np.max(costs), 2),
            "Cost_Median": round(np.median(costs), 2),
            "Cost_Mean": round(np.mean(costs), 2)
        }
    except Exception as e:
        print(f"❌ Error processing {task_name}: {e}")
        return None

if __name__ == "__main__":
    # --- PATH SETUP ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = "/home/20234949/thesis/datasets"
    
    # 1. Find all supported files recursively
    extensions = ['*.hdf5', '*.h5', '*.pkl']
    files_to_process = []
    for ext in extensions:
        files_to_process.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))

    print(f"🚀 Found {len(files_to_process)} datasets in {DATA_DIR}")

    all_stats = []
    for file_path in files_to_process:
        task_name = os.path.splitext(os.path.basename(file_path))[0]
        stats = analyze_dataset(file_path, task_name, SCRIPT_DIR)
        if stats:
            all_stats.append(stats)

    # 2. Save Master CSV
    if all_stats:
        df = pd.DataFrame(all_stats)
        csv_path = os.path.join(SCRIPT_DIR, "master_dataset_stats.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Done! Check {SCRIPT_DIR} for the CSV and PNGs.")
    else:
        print("⚠️ No datasets were successfully processed.")