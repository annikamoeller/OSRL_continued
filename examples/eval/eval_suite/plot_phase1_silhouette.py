#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
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
def crawl_silhouette_data(sweep_dir):
    print(f"🕷️ Crawling directory for metrics: {sweep_dir}")
    search_pattern = os.path.join(sweep_dir, "**", "*metrics_history.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        raise ValueError(f"❌ No metrics_history.csv files found in {sweep_dir}")
        
    records = []
    
    for file_path in csv_files:
        # Example Path: .../OfflineCarCircle-v0/CCDT_Arch_Back_cw10/2B/seed_8_metrics_history.csv
        parts = file_path.split(os.sep)
        
        try:
            # 1. Parse Metadata from Folder Structure
            task_raw = parts[-4]
            task = task_raw.replace("Offline", "").replace("-v0", "")
            
            arch_cw_folder = parts[-3] # e.g., "CCDT_Arch_Back_cw10"
            arch = "Back" if "Back" in arch_cw_folder else "Front"
            cw_str = arch_cw_folder.split("_cw")[-1]
            cw = float(cw_str)
            
            bucket_folder = parts[-2] # e.g., "2B"
            buckets = int(bucket_folder.replace("B", ""))
            
            seed_file = parts[-1] # e.g., "seed_8_metrics_history.csv"
            seed = int(seed_file.split("_")[1])
            
            # 2. Extract Max Silhouette Score
            df = pd.read_csv(file_path)
            if "eval/silhouette_score" in df.columns:
                max_sil = df["eval/silhouette_score"].max()
                records.append({
                    "Task": task,
                    "Architecture": arch,
                    "Buckets": buckets,
                    "CW": cw,
                    "Seed": seed,
                    "Max_Silhouette": max_sil
                })
        except Exception as e:
            print(f"⚠️ Warning: Could not parse {file_path}. Error: {e}")
            continue

    master_df = pd.DataFrame(records)
    
    # Average the maximums across the seeds
    agg_df = master_df.groupby(["Task", "Architecture", "Buckets", "CW"])["Max_Silhouette"].mean().reset_index()
    print(f"✅ Extracted data for {len(agg_df)} unique configurations.")
    return agg_df

# --- PLOTTING ENGINE ---
def plot_cw_vs_silhouette(sweep_dir, output_dir):
    set_professional_style()
    
    sil_data = crawl_silhouette_data(sweep_dir)
    
    # 1. Pre-process and Sort Data
    # Convert CW to categorical so Seaborn treats them as distinct groups
    sil_data["CW"] = sil_data["CW"].astype(str)
    tasks = sorted(sil_data["Task"].unique())
    architectures = sorted(sil_data["Architecture"].unique())
    unique_cws = sorted(sil_data["CW"].unique(), key=float)
    
    # 2. Setup Palette
    palette = sns.color_palette("viridis", n_colors=len(unique_cws))
    
    fig, axes = plt.subplots(nrows=len(architectures), ncols=len(tasks), 
                             figsize=(6 * len(tasks), 5 * len(architectures)), 
                             squeeze=False)
    
    for row_idx, arch in enumerate(architectures):
        for col_idx, task in enumerate(tasks):
            ax = axes[row_idx, col_idx]
            sub_df = sil_data[(sil_data["Architecture"] == arch) & (sil_data["Task"] == task)]
            
            if not sub_df.empty:
                sns.barplot(
                    data=sub_df, x="Buckets", y="Max_Silhouette", hue="CW", 
                    hue_order=unique_cws, palette=palette, 
                    edgecolor="black", linewidth=1.2, ax=ax
                )
            
            # Formatting
            ax.set_title(f"{task} | {arch}" if row_idx == 0 else "", fontweight="bold")
            ax.set_ylabel("Max Silhouette" if col_idx == 0 else "")
            ax.set_xlabel("Number of Buckets")
            
            # Remove individual legends to build one at the end
            if ax.get_legend():
                ax.get_legend().remove()

    #   1. Build the legend handles/labels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # 2. Add the legend to the FIGURE (not the axes)
    # Using fig.legend() is more stable for outside positioning
    fig.legend(
        handles, 
        labels, 
        title="Cost Weight (cw)", 
        loc='center left', 
        bbox_to_anchor=(0.91, 0.5), # Anchor it to the right side
        frameon=True
    )
    
    # 3. CRITICAL: Remove tight_layout() and use manual adjustment
    # This reserves 15% of the right side of the figure for the legend
    plt.subplots_adjust(right=0.85, wspace=0.3, hspace=0.3)
    
    # 4. Save with bbox_inches='tight'
    # This final pass ensures that even if something is slightly off, 
    # it gets clipped correctly into the saved file.
    out_path = os.path.join(output_dir, "phase1_cw_vs_silhouette.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved successfully to {out_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Contrastive Weight vs Max Silhouette Score.")
    parser.add_argument("--sweep_dir", required=True, help="Root directory containing the cluster eval sweeps.")
    parser.add_argument("--out_dir", default=".", help="Directory to save the plot.")
    
    args = parser.parse_args()
    plot_cw_vs_silhouette(args.sweep_dir, args.out_dir)