#!/usr/bin/env python3
import pandas as pd
import numpy as np
import glob
import os
import argparse

def generate_latex_table(csv_dir=".", stats_csv="/home/20234949/thesis/OSRL_continued/dataset_analysis/master_dataset_stats.csv"):
    print(f"🔍 Searching for evaluation CSVs in: {csv_dir}")
    
    # Find all CSVs recursively
    csv_files = glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True)
    
    all_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            # Ensure it's an evaluation dataframe (must have Target_Cost)
            if 'Target_Cost' not in df.columns:
                continue
            all_data.append(df)
        except Exception as e:
            print(f"⚠️ Could not read {file_path}: {e}")

    if not all_data:
        raise ValueError(f"❌ No valid evaluation CSVs found in directory: {csv_dir}")

    master_df = pd.concat(all_data, ignore_index=True)

    # 1. Autodetect and clean column names
    env_col = 'Task' if 'Task' in master_df.columns else 'Environment'
    master_df[env_col] = master_df[env_col].apply(lambda x: x if 'Offline' in x else f"Offline{x}-v0")

    # 2. Dynamically normalize using the local master stats CSV
    if 'Norm_Reward' not in master_df.columns or 'Norm_Cost' not in master_df.columns:
        print(f"🔄 Normalizing metrics using local stats from: {stats_csv}")
        try:
            stats_df = pd.read_csv(stats_csv)
            
            # Build a fast lookup dictionary from your dataset stats
            stats_dict = {}
            for _, row in stats_df.iterrows():
                stats_dict[row['Task']] = {
                    'Return_Max': float(row['Return_Max']),
                    'Return_Min': float(row['Return_Min']),
                    'Cost_Max': float(row['Cost_Max'])
                }
            
            # Helper to map generic env names (OfflineAntRun-v0) to specific dataset names (SafetyAntRun-v0-150...)
            def get_stats(env_name):
                base_name = env_name.replace("Offline", "").replace("-v0", "")
                for st_name, stats in stats_dict.items():
                    if base_name in st_name:
                        return stats
                return None

            def normalize_row(row):
                env_name = row[env_col]
                stats = get_stats(env_name)
                
                rew = row.get('Eval_Reward', row.get('Reward', 0))
                cost = row.get('Eval_Cost', row.get('Cost', 0))
                
                if stats:
                    r_max, r_min, c_max = stats['Return_Max'], stats['Return_Min'], stats['Cost_Max']
                    
                    # Score normalizer (0 = random, 100 = expert max)
                    norm_rew = 100.0 * (rew - r_min) / (r_max - r_min) if (r_max - r_min) > 0 else rew
                    
                    # Cost normalizer (100 = hit the dataset cost limit)
                    norm_cost = 100.0 * (cost / c_max) if c_max > 0 else cost
                else:
                    norm_rew, norm_cost = rew, cost
                    
                return pd.Series([norm_rew, norm_cost])

            master_df[['Norm_Reward', 'Norm_Cost']] = master_df.apply(normalize_row, axis=1)
            
        except Exception as e:
            print(f"⚠️ Failed to normalize using {stats_csv}: {e}")
            print("⚠️ Falling back to raw scores.")
            master_df['Norm_Reward'] = master_df.get('Eval_Reward', master_df.get('Reward'))
            master_df['Norm_Cost'] = master_df.get('Eval_Cost', master_df.get('Cost'))

    rew_col = 'Norm_Reward'
    cost_col = 'Norm_Cost'

    # 3. Handle missing Seed columns (Vanilla might not have it saved in the CSV)
    if 'Seed' not in master_df.columns:
        master_df['Seed'] = 42 
    else:
        master_df['Seed'] = master_df['Seed'].fillna(42)

    # 4. Create a highly descriptive Display Name
    def make_display_name(row):
        arch = str(row.get('Architecture', 'Unknown')).strip()
        if arch.lower() == 'vanilla':
            return 'Vanilla CDT (Baseline)'
        else:
            b = row.get('Buckets', '?')
            cw = row.get('CW', '?')
            return f"{arch} Encoder | {b} Buckets | cw={cw}"

    master_df['Display_Name'] = master_df.apply(make_display_name, axis=1)
    print(f"✅ Loaded {len(master_df)} evaluation data points.")

    # ---------------------------------------------------------
    # STEP 1: Average across the Target Cost spectrum PER SEED
    # ---------------------------------------------------------
    seed_agg = master_df.groupby(
        [env_col, 'Display_Name', 'Architecture', 'Buckets', 'CW', 'Seed']
    )[[rew_col, cost_col]].mean().reset_index()

    # ---------------------------------------------------------
    # STEP 2: Average across SEEDS to get Mean ± Std
    # ---------------------------------------------------------
    final_agg = seed_agg.groupby([env_col, 'Display_Name', 'Architecture', 'Buckets', 'CW']).agg(
        Reward_Mean=(rew_col, 'mean'),
        Reward_Std=(rew_col, 'std'),
        Cost_Mean=(cost_col, 'mean'),
        Cost_Std=(cost_col, 'std')
    ).reset_index()

    # Handle NaNs in std (if only 1 seed is present)
    final_agg.fillna(0.0, inplace=True)

    # Format into strings (e.g. 70.1 ± 2.5)
    final_agg['Formatted_Reward'] = final_agg.apply(lambda r: f"{r['Reward_Mean']:.1f} ± {r['Reward_Std']:.1f}", axis=1)
    final_agg['Formatted_Cost'] = final_agg.apply(lambda r: f"{r['Cost_Mean']:.1f} ± {r['Cost_Std']:.1f}", axis=1)
    
    final_agg['Combined'] = final_agg['Formatted_Reward'] + "  |  " + final_agg['Formatted_Cost']

    # ---------------------------------------------------------
    # STEP 3: Smart Sorting (Baseline First, then Group by Arch, Buckets, CW)
    # ---------------------------------------------------------
    final_agg['Sort_Key_1'] = final_agg['Architecture'].apply(lambda x: 0 if str(x).lower() == 'vanilla' else 1)
    final_agg['Sort_Key_2'] = final_agg['Architecture'].astype(str)
    final_agg['Sort_Key_3'] = pd.to_numeric(final_agg['Buckets'], errors='coerce').fillna(0)
    final_agg['Sort_Key_4'] = pd.to_numeric(final_agg['CW'], errors='coerce').fillna(0)
    
    final_agg = final_agg.sort_values(by=['Sort_Key_1', 'Sort_Key_2', 'Sort_Key_3', 'Sort_Key_4'])

    # ---------------------------------------------------------
    # STEP 4: Pivot Table for LaTeX
    # ---------------------------------------------------------
    pivot_df = final_agg.pivot(index='Display_Name', columns=env_col, values='Combined')
    
    # Re-apply the sorted order to the pivot index
    ordered_index = final_agg['Display_Name'].drop_duplicates().tolist()
    pivot_df = pivot_df.reindex(ordered_index)
    
    # Print Markdown version to terminal
    print("\n" + "="*85)
    print(" 📊 AGGREGATED METRICS (Norm. Reward  |  Norm. Cost)")
    print("="*85)
    print(pivot_df.to_markdown())
    print("="*85 + "\n")

    # Generate LaTeX
    latex_str = pivot_df.to_latex(column_format="l" + "c" * len(pivot_df.columns), escape=False)
    latex_str = latex_str.replace("±", r"$\pm$") 
    
    # Add a publication-ready top rule and header
    latex_str = latex_str.replace("\\toprule", "\\toprule\n\\multirow{2}{*}{\\textbf{Architecture Variant}} & \\multicolumn{" + str(len(pivot_df.columns)) + "}{c}{\\textbf{Environment (Norm. Reward | Norm. Cost)}} \\\\\n\\cmidrule{2-" + str(len(pivot_df.columns)+1) + "}")
    
    out_file = "aggregated_metrics_comparison.tex"
    with open(out_file, "w") as f:
        f.write(latex_str)
        
    print(f"✅ Publication-ready LaTeX table saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Aggregated Reward/Cost Tables")
    parser.add_argument("--csv_dir", default=".", help="Directory containing your evaluation CSV files")
    parser.add_argument("--stats_csv", default="/home/20234949/thesis/OSRL_continued/dataset_analysis/master_dataset_stats.csv", help="Path to your dataset stats CSV")
    args = parser.parse_args()
    
    generate_latex_table(args.csv_dir, args.stats_csv)