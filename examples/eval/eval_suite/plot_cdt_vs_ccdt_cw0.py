#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --- PROFESSIONAL THESIS STYLING ---
def set_thesis_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
        }
    )


# Standard color palette for the whole thesis
THESIS_PALETTE = {
    "Vanilla Baseline": "#7f7f7f",  # Neutral Grey
    "CCDT Ablation (No Contrastive)": "#1f77b4",  # Muted Blue
}


def generate_targeted_grid_plots(vanilla_path, ablation_path, output_dir):
    # Apply global style to matplotlib
    set_thesis_style()

    if not os.path.exists(vanilla_path) or not os.path.exists(ablation_path):
        print(f"❌ Error: Missing source data spreadsheet components.")
        print(f"Checked Vanilla: {vanilla_path}")
        print(f"Checked Ablation: {ablation_path}")
        return

    # Load baseline assets
    print("📂 Loading CSV data...")
    df_cdt = pd.read_csv(vanilla_path)
    df_ablation = pd.read_csv(ablation_path)

    # Harmonize model identification labels
    df_cdt["Model_Variant"] = "Vanilla Baseline"
    df_ablation["Model_Variant"] = "CCDT Ablation (No Contrastive)"

    # Fill structural layout properties for the ablation matrix
    df_ablation["Architecture"] = "Front"
    df_ablation["Buckets"] = "1 Buckets"
    df_ablation["Contrastive_Weight"] = 0.0

    # Harmonize task string identifiers
    for df in [df_cdt, df_ablation]:
        if "Task" in df.columns:
            df["Clean_Task"] = df["Task"].astype(str).str.replace("Offline", "").str.replace("-v0", "")
        else:
            df["Clean_Task"] = "Environment"

    # Pool tracking matrices safely
    global_df = pd.concat([df_cdt, df_ablation], ignore_index=True)

    # 🌟 FILTER STAGE: Restrict processing exclusively to targeted tasks
    TARGET_TASKS = ["AntRun", "CarCircle", "DroneRun"]
    global_df = global_df[global_df["Clean_Task"].isin(TARGET_TASKS)]

    # Enforce numerical formatting
    global_df["Target_Cost"] = pd.to_numeric(global_df["Target_Cost"])
    global_df["Raw_Eval_Cost"] = pd.to_numeric(global_df["Raw_Eval_Cost"])
    global_df["Raw_Eval_Reward"] = pd.to_numeric(global_df["Raw_Eval_Reward"])

    ordered_variants = ["CCDT Ablation (No Contrastive)", "Vanilla Baseline"]

    # Loop through only the targeted environments
    unique_tasks = sorted(global_df["Clean_Task"].unique())
    print(f"📊 Discovered targeted tasks for processing: {unique_tasks}")

    for task_name in unique_tasks:
        task_df = global_df[global_df["Clean_Task"] == task_name]

        # Initialize the matching 2x1 grid panel
        fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

        # --- ROW 1: REWARD RETURNS ---
        sns.lineplot(
            ax=axes[0],
            data=task_df,
            x="Target_Cost",
            y="Raw_Eval_Reward",
            hue="Model_Variant",
            style="Model_Variant",
            palette=THESIS_PALETTE,
            hue_order=ordered_variants,
            style_order=ordered_variants,
            markers=True,
            markersize=8,
            dashes=False,
            linewidth=2,
            errorbar="sd",
        )
        axes[0].set_title(f"{task_name} (Ablation Analysis)", fontweight="bold")
        axes[0].set_ylabel("Evaluated Episode Return")
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        # Suppress the internal legend
        if axes[0].get_legend() is not None:
            axes[0].get_legend().remove()

        # --- ROW 2: ACTUAL EVALUATED COST ADHERENCE ---
        sns.lineplot(
            ax=axes[1],
            data=task_df,
            x="Target_Cost",
            y="Raw_Eval_Cost",
            hue="Model_Variant",
            style="Model_Variant",
            palette=THESIS_PALETTE,
            hue_order=ordered_variants,
            style_order=ordered_variants,
            markers=True,
            markersize=8,
            dashes=False,
            linewidth=2,
            errorbar="sd",
        )

        # Ideal line trend projection
        sweep_vals = sorted(task_df["Target_Cost"].unique())
        axes[1].plot(sweep_vals, sweep_vals, "k:", alpha=0.6, label="Ideal Adherence (y=x)")
        axes[1].set_ylabel("Actual Evaluated Cost")
        axes[1].set_xlabel("Target Cost Prompt")
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        if axes[1].get_legend() is not None:
            axes[1].get_legend().remove()

        # --- GLOBAL LEGEND ---
        handles1, labels1 = axes[0].get_legend_handles_labels()
        handles2, labels2 = axes[1].get_legend_handles_labels()

        all_handles = handles1 + [h for h, l in zip(handles2, labels2) if l == "Ideal Adherence (y=x)"]
        all_labels = labels1 + [l for l in labels2 if l == "Ideal Adherence (y=x)"]

        fig.legend(all_handles, all_labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

        # Adjust layout to make room for the bottom legend
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # --- SAVE IMAGES LOCALLY ---
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"grid_ablation_{task_name.lower()}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free up memory

        print(f"✅ Target grid successfully generated for: {task_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CCDT vs Vanilla Ablation Grids")

    # Default paths are set to your local directories
    parser.add_argument(
        "--vanilla_path",
        type=str,
        default="/Users/annikamollerchandiramani/Documents/uni/OSRL_continued/examples/eval/eval_suite/eval_vanilla_cdt/raw_vanilla_data.csv",
        help="Path to the Vanilla CDT CSV data",
    )
    parser.add_argument(
        "--ablation_path",
        type=str,
        default="/Users/annikamollerchandiramani/Documents/uni/OSRL_continued/examples/eval/eval_suite/vanilla_csv_results_ccdt/raw_data.csv",
        help="Path to the CCDT Ablation CSV data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/annikamollerchandiramani/Documents/uni/OSRL_continued/examples/eval/eval_suite/plots",
        help="Directory to save the output PDFs",
    )

    args = parser.parse_args()

    generate_targeted_grid_plots(args.vanilla_path, args.ablation_path, args.output_dir)
