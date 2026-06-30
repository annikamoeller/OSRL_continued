import pandas as pd
import os
import argparse

def isolate_data(master_csv, target_cw, target_arch, out_dir):
    print(f"📥 Loading master data from {master_csv}...")
    df = pd.read_csv(master_csv)
    
    # Handle slight variations in your CSV headers
    cw_col = "CW" if "CW" in df.columns else "Contrastive_Weight"
    
    # Apply the filters
    print(f"✂️ Filtering for {target_arch} Architecture and CW = {target_cw}...")
    df_filtered = df[(df[cw_col] == target_cw) & (df["Architecture"] == target_arch)]
    
    if df_filtered.empty:
        print("❌ Error: No data found matching those filters!")
        return
        
    # Create the new directory and save
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "raw_data.csv")
    df_filtered.to_csv(out_path, index=False)
    
    print(f"✅ Extracted {len(df_filtered)} rows.")
    print(f"📁 Saved isolated data to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to your master raw_data.csv")
    parser.add_argument("--cw", type=float, default=0.5, help="Target Contrastive Weight")
    parser.add_argument("--arch", type=str, default="Back", help="Target Architecture (Front/Back)")
    parser.add_argument("--out", required=True, help="New directory to save the filtered CSV")
    
    args = parser.parse_args()
    isolate_data(args.csv, args.cw, args.arch, args.out)