#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import argparse

PROJECT_ROOT = "/home/20234949/thesis/OSRL_continued"
sys.path.insert(0, PROJECT_ROOT)

# Master configuration
STATS_CSV = os.path.join(PROJECT_ROOT, "dataset_analysis", "master_dataset_stats.csv")
EPSILON = 1e-8


def generate_combined_publication_table(vanilla_path, ablation_path, output_dir, target_cost=10.0):
    if not os.path.exists(vanilla_path) or not os.path.exists(ablation_path):
        print("❌ Error: Missing source data CSVs.")
        return
    if not os.path.exists(STATS_CSV):
        print(f"❌ Error: Missing master stats at {STATS_CSV}")
        return

    # 1. Load Data
    print("📂 Loading Vanilla and Ablation CSVs...")
    df_cdt = pd.read_csv(vanilla_path)
    df_ablation = pd.read_csv(ablation_path)

    stats_df = pd.read_csv(STATS_CSV)
    stats_lookup = stats_df.set_index("Task").to_dict("index")

    # 2. Harmonize Variants
    df_cdt["Variant"] = "Vanilla Baseline"
    df_ablation["Variant"] = "CCDT Ablation (No Contrastive)"

    # 3. Combine DataFrames
    global_df = pd.concat([df_cdt, df_ablation], ignore_index=True)

    # Harmonize Task names
    if "Task" in global_df.columns:
        global_df["Clean_Task"] = global_df["Task"].astype(str).str.replace("Offline", "").str.replace("-v0", "")
    else:
        print("❌ Error: 'Task' column missing from data.")
        return

    # 4. Filter for target cost (Standard is 10.0)
    table_data = global_df[global_df["Target_Cost"] == target_cost].copy()

    if table_data.empty:
        print(f"⚠️ No data found for Target Cost = {target_cost}.")
        return

    processed_records = []

    # 5. Apply Normalization Math
    for _, row in table_data.iterrows():
        task_name = row["Clean_Task"]
        match = next((k for k in stats_lookup.keys() if task_name in k), None)

        if match:
            r_max = stats_lookup[match]["Return_Max"]
            r_min = stats_lookup[match]["Return_Min"]
        else:
            r_max, r_min = 1000.0, 0.0

        # Reward Normalization
        norm_reward = ((row["Raw_Eval_Reward"] - r_min) / (r_max - r_min + EPSILON)) * 100

        # Cost Normalization
        norm_cost = row["Raw_Eval_Cost"] / (row["Target_Cost"] + EPSILON)

        processed_records.append(
            {"Task": task_name, "Variant": row["Variant"], "Norm_Reward": norm_reward, "Norm_Cost": norm_cost}
        )

    eval_df = pd.DataFrame(processed_records)

    # 6. Group by Task and Variant, calculate Mean and Std Dev
    summary = (
        eval_df.groupby(["Task", "Variant"])
        .agg({"Norm_Reward": ["mean", "std"], "Norm_Cost": ["mean", "std"]})
        .round(2)
    )

    # Flatten multi-level columns
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]

    # 7. Save and Print
    os.makedirs(output_dir, exist_ok=True)
    output_table_csv = os.path.join(output_dir, f"combined_publication_table_kappa{int(target_cost)}.csv")
    summary.to_csv(output_table_csv)

    print("\n" + "═" * 70)
    print(f"📊 COMBINED PUBLICATION SUMMARY TABLE (Target Cost κ = {target_cost})")
    print("═" * 70)
    print(summary.to_string())
    print("═" * 70)
    print(f"✅ Table saved to: {output_table_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Combined Publication Tables")

    # Defaults set to your local MacOS paths from the previous plotting script
    parser.add_argument(
        "--vanilla_path",
        type=str,
        default="examples/eval/eval_suite/eval_vanilla_cdt/raw_vanilla_data.csv",
        help="Path to the Vanilla CDT CSV",
    )
    parser.add_argument(
        "--ablation_path",
        type=str,
        default="examples/eval/eval_suite/vanilla_csv_results_ccdt/raw_data.csv",
        help="Path to the CCDT Ablation CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="examples/eval/eval_suite",
        help="Directory to save the CSV table",
    )
    parser.add_argument("--target_cost", type=float, default=10.0, help="The Target Cost (kappa) to filter for")

    args = parser.parse_args()

    generate_combined_publication_table(args.vanilla_path, args.ablation_path, args.output_dir, args.target_cost)