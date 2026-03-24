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
    print(f"🔎 Scanning: {task_name}...")
    
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
        fig.suptitle(f"Dataset: {task_name} | {len(trajectories)} Trajs", fontsize=16, weight='bold')

        sns.histplot(returns, color="blue", alpha=0.4, label="Returns", kde=True, ax=axes[0])
        sns.histplot(costs, color="red", alpha=0.4, label="Costs", kde=True, ax=axes[0])
        axes[0].set_title("Reward/Cost Distribution")
        axes[0].legend()

        sns.scatterplot(x=costs, y=returns, color="purple", alpha=0.5, ax=axes[1])
        axes[1].set_title("Return vs. Cost Pareto")
        axes[1].set_xlabel("Episode Cost")
        axes[1].set_ylabel("Episode Return")

        max_len = np.max(lengths)
        cum_r = np.full((len(trajectories), max_len), np.nan)
        cum_c = np.full((len(trajectories), max_len), np.nan)
        for i, t in enumerate(trajectories):
            cum_r[i, :t["length"]] = np.cumsum(t["rewards"])
            cum_c[i, :t["length"]] = np.cumsum(t["costs"])
        
        axes[2].plot(np.nanmean(cum_r, axis=0), color="blue", label="Mean Cum. Reward")
        axes[2].plot(np.nanmean(cum_c, axis=0), color="red", label="Mean Cum. Cost")
        axes[2].set_title("Avg Accumulation over Time")
        axes[2].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"analysis_{task_name}.png"), dpi=300)
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
        # Generate task name from filename (e.g., 'AntRun_expert.hdf5' -> 'AntRun_expert')
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