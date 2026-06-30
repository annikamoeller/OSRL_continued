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
def calculate_tcva(df, is_vanilla=False):
    """Calculates Total Cost Violation Area (TCVA) using trapezoidal integration."""
    if "Raw_Eval_Cost" in df.columns:
        df = df.rename(columns={"Raw_Eval_Cost": "Eval_Cost"})
    if "Contrastive_Weight" in df.columns:
        df = df.rename(columns={"Contrastive_Weight": "CW"})
        
    # 1. Average the costs across seeds for each Target Cost step first
    keys_for_mean = ["Task", "Target_Cost"] if is_vanilla else ["Task", "Architecture", "Buckets", "CW", "Target_Cost"]
    mean_costs = df.groupby(keys_for_mean)["Eval_Cost"].mean().reset_index()
    
    # 2. Calculate the Overshoot exactly as defined in the LaTeX
    mean_costs["Overshoot"] = np.maximum(0, mean_costs["Eval_Cost"] - mean_costs["Target_Cost"])
    
    # 3. Integrate the area using the Trapezoidal Rule
    group_keys = ["Task"] if is_vanilla else ["Task", "Architecture", "Buckets", "CW"]
    results = []
    
    for key_vals, group in mean_costs.groupby(group_keys):
        # Ensure data is sorted by X-axis (Target Cost) before integrating
        group = group.sort_values("Target_Cost")
        
        # np.trapz(y, x)
        tcva = np.trapz(group["Overshoot"], group["Target_Cost"])
        
        # Reconstruct the row dict
        row = dict(zip(group_keys, [key_vals] if isinstance(key_vals, str) else key_vals))
        row["TCVA"] = tcva
        results.append(row)
        
    return pd.DataFrame(results)

# --- PLOTTING ENGINE ---
def plot_cw_vs_adherence(ccdt_csv, vanilla_csv, output_dir):
    set_professional_style()
    
    # 1. Load and process data
    print("📥 Loading and calculating Adherence Scores (TCVA)...")
    df_ccdt = pd.read_csv(ccdt_csv)
    df_vanilla = pd.read_csv(vanilla_csv)
    
    tcva_ccdt = calculate_tcva(df_ccdt, is_vanilla=False)
    tcva_vanilla = calculate_tcva(df_vanilla, is_vanilla=True)
    
    tasks = sorted(tcva_ccdt["Task"].unique())
    
    # --- FIX 1: DYNAMIC ARCHITECTURES ---
    architectures = sorted(tcva_ccdt["Architecture"].unique())
    num_rows = len(architectures)
    
    # Adjust figure size based on how many rows we actually need
    fig, axes = plt.subplots(nrows=num_rows, ncols=len(tasks), figsize=(5 * len(tasks), 5 * num_rows), sharex=True)
    
    # Safely handle 1D axes arrays
    if num_rows == 1 and len(tasks) > 1:
        axes = np.expand_dims(axes, axis=0)
    elif num_rows > 1 and len(tasks) == 1:
        axes = np.expand_dims(axes, axis=1)
    elif num_rows == 1 and len(tasks) == 1:
        axes = np.array([[axes]])
        
    for row_idx, arch in enumerate(architectures):
        for col_idx, task in enumerate(tasks):
            ax = axes[row_idx, col_idx]
            sub_df = tcva_ccdt[(tcva_ccdt["Architecture"] == arch) & (tcva_ccdt["Task"] == task)]
            
            if sub_df.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, color='gray')
            else:
                # --- FIXED: Added ax=ax so it draws in the correct grid box ---
                sns.barplot(
                    data=sub_df, x="Buckets", y="TCVA", hue="CW", 
                    palette=sns.color_palette("viridis", n_colors=len(tcva_ccdt["CW"].unique())),
                    edgecolor="black", linewidth=1.5, ax=ax
                )
            
            # --- VANILLA BASELINE OVERLAY ---
            v_baseline = tcva_vanilla[tcva_vanilla["Task"] == task]["TCVA"].values
            if len(v_baseline) > 0:
                ax.axhline(v_baseline[0], color="#E63946", linestyle="--", linewidth=2.5, zorder=5, label="Vanilla Baseline")
            
            # Formatting
            if row_idx == 0:
                ax.set_title(f"{task}", fontweight="bold", pad=15)
            if col_idx == 0:
                ax.set_ylabel(f"{arch} Encoder\nAdherence Error (TCVA)", fontweight="bold")
            else:
                ax.set_ylabel("")
                
            # --- FIXED: X-axis is now Buckets ---
            if row_idx == num_rows - 1:
                ax.set_xlabel("Number of Buckets", fontweight="bold")
            else:
                ax.set_xlabel("")

    # --- GLOBAL LEGEND CLEANUP ---
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        if l:
            handles, labels = h, l
        if ax.get_legend() is not None:
            ax.get_legend().remove()
            
    if labels:
        # --- FIXED: Legend title is now Cost Weight ---
        if not str(labels[0]).isalpha(): 
            labels.insert(0, "Cost Weight (cw)")
            handles.insert(0, handles[0]) 
        else:
            labels[0] = "Cost Weight (cw)"
        
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