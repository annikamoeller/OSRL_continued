#!/usr/bin/env python3
import os
import glob
import pandas as pd
import argparse

def standardize_csvs(root_dir, backup=True):
    print(f"🧹 Scanning for CSVs in: {root_dir}")
    
    # Recursively find all CSV files
    search_pattern = os.path.join(root_dir, "**", "*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print("❌ No CSV files found in the target directory!")
        return

    # --- THE STANDARD FORMAT PROTOCOL ---
    column_mapping = {
        "Raw_Eval_Cost": "Eval_Cost",
        "Raw_Eval_Reward": "Eval_Reward",
        "Contrastive_Weight": "CW"
    }

    variant_mapping = {
        "Back - 2 Buckets": "Back-2B",
        "Back - 3 Buckets": "Back-3B",
        "Back - 5 Buckets": "Back-5B",
        "Front - 2 Buckets": "Front-2B",
        "Front - 3 Buckets": "Front-3B",
        "Front - 5 Buckets": "Front-5B",
        "CDT": "CDT Baseline",
        "Vanilla": "Vanilla Baseline"
    }

    files_modified = 0

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            changed = False
            
            # 1. Standardize Headers
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
                    changed = True
            
            # 2. Standardize Variant Naming
            if "Variant" in df.columns:
                # Only apply if there are actually messy names to fix
                if df["Variant"].isin(variant_mapping.keys()).any():
                    df["Variant"] = df["Variant"].replace(variant_mapping)
                    changed = True

            # 3. Save the clean data
            if changed:
                if backup:
                    backup_path = file_path + ".bak"
                    # Don't overwrite an existing backup
                    if not os.path.exists(backup_path):
                        os.rename(file_path, backup_path)
                    else:
                        print(f"⚠️ Backup already exists for {file_path}. Skipping safety rename.")
                
                df.to_csv(file_path, index=False)
                print(f"✅ Cleaned: {os.path.basename(file_path)}")
                files_modified += 1
                
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")

    print(f"\n🎉 Done! Permanently standardized {files_modified} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize CSV column headers and variant names.")
    parser.add_argument("--dir", required=True, help="Directory to recursively scan for CSVs")
    parser.add_argument("--no-backup", action="store_false", dest="backup", help="Skip creating .bak backup files")
    args = parser.parse_args()
    
    standardize_csvs(args.dir, args.backup)