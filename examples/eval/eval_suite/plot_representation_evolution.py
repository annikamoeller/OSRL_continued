#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# --- PUBLICATION STYLING ---
def set_professional_style():
    sns.set_theme(style="whitegrid", font="sans-serif", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.2, "grid.linestyle": "--", "grid.alpha": 0.7
    })

# --- DATA CRAWLER ---
def crawl_timeseries_data(sweep_dir, target_arch, target_cw):
    print(f"🕷️ Crawling directory for {target_arch} Encoder at cw={target_cw}: {sweep_dir}")
    search_pattern = os.path.join(sweep_dir, "**", "*metrics_history.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        raise ValueError(f"❌ No metrics_history.csv files found in {sweep_dir}")
        
    all_data = []
    
    for file_path in csv_files:
        parts = file_path.split(os.sep)
        try:
            # 1. Parse Metadata
            task_raw = parts[-4]
            task = task_raw.replace("Offline", "").replace("-v0", "")
            
            arch_cw_folder = parts[-3]
            arch = "Back" if "Back" in arch_cw_folder else "Front"
            cw = float(arch_cw_folder.split("_cw")[-1])
            
            buckets = int(parts[-2].replace("B", ""))
            seed = int(parts[-1].split("_")[1])
            
            # 2. Filter for the target architecture and specific Contrastive Weight
            if arch != target_arch or cw != target_cw:
                continue
                
            # 3. Read and Standardize
            df = pd.read_csv(file_path)
            
            col_map = {
                "_step": "Step", "step": "Step",
                "eval/silhouette_score": "Silhouette",
                "eval/linear_probe_score": "LinearProbe"
            }
            df = df.rename(columns=col_map)
            
            df["Task"] = task
            df["Buckets"] = buckets
            df["Seed"] = seed
            
            keep_cols = ["Task", "Buckets", "Seed", "Step", "Silhouette", "LinearProbe"]
            df = df[[c for c in keep_cols if c in df.columns]]
            
            all_data.append(df)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not parse {file_path}. Error: {e}")
            continue

    if not all_data:
        raise ValueError(f"❌ No data matched Architecture='{target_arch}' and CW='{target_cw}'")

    master_df = pd.concat(all_data, ignore_index=True)
    # Ensure Buckets are treated as categories for clean plotting and legend sorting
    master_df["Buckets"] = master_df["Buckets"].astype(str) + " Buckets" 
    print(f"✅ Extracted {len(master_df)} time-series steps across all environments.")
    return master_df

# --- PLOTTING ENGINE ---
def plot_bucket_comparison(df, metric, output_dir, target_arch, target_cw):
    tasks = sorted(df["Task"].unique())
    
    # Use standard publication colors for the buckets
    color_map = {
        "2 Buckets": "#1f77b4", # Blue
        "3 Buckets": "#2ca02c", # Green
        "5 Buckets": "#ff7f0e", # Orange
    }
    
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), sharex=True, sharey=True)
    if len(tasks) == 1: axes = [axes]
        
    for col_idx, task in enumerate(tasks):
        ax = axes[col_idx]
        sub_df = df[df["Task"] == task].sort_values("Buckets")
        
        sns.lineplot(
            data=sub_df, x="Step", y=metric, hue="Buckets", 
            palette=color_map, linewidth=2.0, ax=ax,
            legend=(col_idx == len(tasks)-1)
        )
        
        # Formatting
        ax.set_title(f"{task}", fontweight="bold", pad=15)
        ax.set_xlabel("Training Steps", fontweight="bold")
        
        if col_idx == 0:
            metric_name = "Silhouette Score" if metric == "Silhouette" else "Linear Probe Accuracy"
            ax.set_ylabel(metric_name, fontweight="bold")
            
    # Fix Master Legend
    handles, labels = axes[-1].get_legend_handles_labels()
    if labels:
        if labels[0] == "Buckets": 
            handles = handles[1:]
            labels = labels[1:]
        axes[-1].legend(handles, labels, title="Architecture", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    plt.suptitle(f"{metric} Evolution | {target_arch} Encoder (cw={target_cw})", y=1.05, weight='bold', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"bucket_comparison_{metric.lower()}_{target_arch.lower()}_cw{target_cw}.png"
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ {metric} Plot saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", required=True, help="Root directory containing the cluster eval sweeps.")
    parser.add_argument("--out_dir", default=".", help="Directory to save the plots.")
    parser.add_argument("--arch", default="Back", help="Which architecture to plot (Front/Back).")
    parser.add_argument("--cw", type=float, default=50.0, help="Which Contrastive Weight to isolate.")
    
    args = parser.parse_args()
    set_professional_style()
    
    df = crawl_timeseries_data(args.sweep_dir, args.arch, args.cw)
    
    if "Silhouette" in df.columns:
        plot_bucket_comparison(df, "Silhouette", args.out_dir, args.arch, args.cw)
    if "LinearProbe" in df.columns:
        plot_bucket_comparison(df, "LinearProbe", args.out_dir, args.arch, args.cw)