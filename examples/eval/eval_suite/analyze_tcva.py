#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import os

def calculate_tcva(df):
    """
    Calculates TCVA grouped by your experimental settings, 
    specifically exposing the 'Buckets' column.
    """
    # Standardize column names if needed
    if "Raw_Eval_Cost" in df.columns:
        df = df.rename(columns={"Raw_Eval_Cost": "Eval_Cost"})
        
    # Grouping ensures we get a TCVA score for every unique configuration
    # including the number of buckets
    group_keys = ["Task", "Architecture", "Buckets", "CW"]
    
    results = []
    
    # Group by config
    for key_vals, group in df.groupby(group_keys):
        # 1. Average seeds for this specific config and target cost
        # We need mean cost at every target cost step
        grouped_by_target = group.groupby("Target_Cost")["Eval_Cost"].mean().reset_index()
        
        # 2. Calculate Overshoot: max(0, cost - target)
        grouped_by_target["Overshoot"] = np.maximum(0, grouped_by_target["Eval_Cost"] - grouped_by_target["Target_Cost"])
        
        # 3. Sort by target cost (essential for integration)
        grouped_by_target = grouped_by_target.sort_values("Target_Cost")
        
        # 4. Integrate using Trapezoidal Rule
        tcva = np.trapz(grouped_by_target["Overshoot"], grouped_by_target["Target_Cost"])
        
        # Store result
        row = dict(zip(group_keys, key_vals))
        row["TCVA"] = tcva
        results.append(row)
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to your raw_data.csv")
    parser.add_argument("--out", default="tcva_bucket_analysis.csv", help="Output filename")
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    tcva_df = calculate_tcva(df)
    
    # Save to CSV for your report
    tcva_df.to_csv(args.out, index=False)
    print(f"✅ TCVA analysis complete. Table saved to {args.out}")
    print("\n--- Summary Table ---")
    print(tcva_df.sort_values(["Task", "Buckets"]))