import wandb
import pandas as pd
import os

# Configuration
ENTITY = "annika-moeller24-eindhoven-university-of-technology"
PROJECT = "CCDT_Test_Runs"
METRIC_KEY = "eval/linear_probe_r2"
OUTPUT_DIR = "downloaded_metrics"

def download_metrics():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Found {len(runs)} runs in project {PROJECT}...")

    for run in runs:
        # Fetch history for the specific key
        history = run.history(keys=[METRIC_KEY, "_step"], samples=1000)
        
        if METRIC_KEY in history.columns:
            # Add run metadata
            history['run_name'] = run.name
            history['run_id'] = run.id
            
            # Save to CSV
            file_path = os.path.join(OUTPUT_DIR, f"{run.name}_{run.id}.csv")
            history.to_csv(file_path, index=False)
            print(f"Downloaded: {run.name}")
        else:
            print(f"Skipped (Metric not found): {run.name}")

if __name__ == "__main__":
    download_metrics()