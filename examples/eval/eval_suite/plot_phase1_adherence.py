#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- PUBLICATION STYLING ---
def set_professional_style():
    sns.set_theme(style="whitegrid", font="sans-serif", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.2, "grid.linestyle": "--", "grid.alpha": 0.7
    })

# --- DATA PROCESSING ---
def calculate_mae(df, is_vanilla=False):
    """Calculates Mean Absolute Error for Target vs Evaluated Cost."""
    # Unify column names between CCDT and Vanilla CSVs
    if "Raw_Eval_Cost" in df.columns:
        df = df.rename(columns={"Raw_Eval_Cost": "Eval_Cost"})
    if "Contrastive_Weight" in df.columns:
        df = df.rename(columns={"Contrastive_Weight": "CW"})
        
    df["MAE"] = np.abs(df["Target_Cost"] - df["Eval_Cost"])
    
    if is_vanilla:
        # Vanilla just needs the average MAE per Task
        return df.groupby(["Task"])["MAE"].mean().reset_index()
    else:
        # CCDT needs average MAE per Task, Architecture, Buckets, and CW
        # We average over Seeds and Target_Costs
        return df.groupby(["Task", "Architecture", "Buckets", "CW"])["MAE"].mean().reset_index()

# --- PLOTTING ENGINE ---
def plot_cw_vs_adherence(ccdt_csv, vanilla_csv, output_dir):
    set_professional_style()
    
    # 1. Load and process data
    print("📥 Loading and calculating Adherence Scores (MAE)...")
    df_ccdt = pd.read_csv(ccdt_csv)
    df_vanilla = pd.read_csv(vanilla_csv)
    
    mae_ccdt = calculate_mae(df_ccdt, is_vanilla=False)
    mae_vanilla = calculate_mae(df_vanilla, is_vanilla=True)
    
    tasks = sorted(mae_ccdt["Task"].unique())
    
    # --- FIX 1: DYNAMIC ARCHITECTURES ---
    # Automatically detects if you only have "Back", or both "Front" and "Back"
    architectures = sorted(mae_ccdt["Architecture"].unique())
    num_rows = len(architectures)
    
    palette = sns.color_palette("viridis", n_colors=len(mae_ccdt["Buckets"].unique()))
    
    # Adjust figure size based on how many rows we actually need
    fig, axes = plt.subplots(nrows=num_rows, ncols=len(tasks), figsize=(5 * len(tasks), 5 * num_rows), sharex=True)
    
    # Safely handle 1D axes arrays if there's only 1 row or 1 column
    if num_rows == 1 and len(tasks) > 1:
        axes = np.expand_dims(axes, axis=0)
    elif num_rows > 1 and len(tasks) == 1:
        axes = np.expand_dims(axes, axis=1)
    elif num_rows == 1 and len(tasks) == 1:
        axes = np.array([[axes]])
        
    for row_idx, arch in enumerate(architectures):
        for col_idx, task in enumerate(tasks):
            ax = axes[row_idx, col_idx]
            sub_df = mae_ccdt[(mae_ccdt["Architecture"] == arch) & (mae_ccdt["Task"] == task)]
            
            # --- FIX 2: ALWAYS DRAW VANILLA BASELINE ---
            v_baseline = mae_vanilla[mae_vanilla["Task"] == task]["MAE"].values
            if len(v_baseline) > 0:
                ax.axhline(v_baseline[0], color="#E63946", linestyle="--", linewidth=2, label="Vanilla Baseline")
            
            if sub_df.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, color='gray')
            else:
                sns.lineplot(
                    data=sub_df, x="CW", y="MAE", hue="Buckets", 
                    palette=palette, marker="o", linewidth=2.5, ax=ax, 
                    legend=True # Let Seaborn draw legends normally for now
                )
            
            # Formatting
            if row_idx == 0:
                ax.set_title(f"{task}", fontweight="bold", pad=15)
            if col_idx == 0:
                ax.set_ylabel(f"{arch} Encoder\nAdherence Error (MAE)", fontweight="bold")
            else:
                ax.set_ylabel("")
                
            if row_idx == num_rows - 1:
                ax.set_xlabel("Contrastive Weight (cw)", fontweight="bold")
            else:
                ax.set_xlabel("")

    # --- FIX 3: GLOBAL LEGEND CLEANUP ---
    # Grab the handles from ANY valid axis, then hide all individual legends
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        if l:
            handles, labels = h, l
        if ax.get_legend() is not None:
            ax.get_legend().remove()
            
    if labels:
        if not str(labels[0]).isalpha(): 
            labels.insert(0, "Buckets")
            handles.insert(0, handles[0]) 
        else:
            labels[0] = "Buckets"
        
        # Place one master legend on the far right
        axes[0, -1].legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
        
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "phase1_cw_vs_adherence.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Plot successfully saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Contrastive Weight vs Cost Adherence.")
    parser.add_argument("--ccdt_csv", required=True, help="Path to raw_data.csv for CCDT Bucket sweeps.")
    parser.add_argument("--vanilla_csv", required=True, help="Path to raw_data.csv for Vanilla baseline.")
    parser.add_argument("--out_dir", default=".", help="Directory to save the plot.")
    
    args = parser.parse_args()
    plot_cw_vs_adherence(args.ccdt_csv, args.vanilla_csv, args.out_dir)