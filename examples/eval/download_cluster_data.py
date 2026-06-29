#!/usr/bin/env python3
import os
import pandas as pd
import wandb

def download_numerical_thesis_cache(
    entity: str,
    project_names: list,
    base_save_dir: str,
    silhouette_keyword: str = "silhouette",
    probe_keyword: str = "probe",
):
    """
    Downloads numerical evaluation metrics for the 6 most recent finished runs
    across the specified W&B projects.
    """
    api = wandb.Api()

    for project_name in project_names:
        print(f"\n🚀 Processing project: {project_name}")
        
        try:
            # Query runs: filter by finished, sort by newest first, take top 6
            runs = api.runs(
                f"{entity}/{project_name}", 
                filters={"state": "finished"},
                order="-created_at"
            )[:6]
        except Exception as e:
            print(f"❌ Failed to access project {project_name}: {e}")
            continue

        total_runs = len(runs)
        print(f"✅ Found {total_runs} recent runs to process.")

        for idx, run in enumerate(runs):
            cfg = run.config
            env = cfg.get("task", "unknown_task")
            buckets = cfg.get("num_buckets", 1)
            seed_val = cfg.get("seed", 0)

            # Standardize path components
            try:
                buckets = int(float(buckets))
                seed_val = int(float(seed_val))
            except (ValueError, TypeError):
                buckets, seed_val = 1, 0

            print(f"   📥 [{idx+1}/{total_runs}] Syncing: {run.name} (Seed {seed_val})")

            # Path structure
            run_save_dir = os.path.join(base_save_dir, project_name, f"{buckets}B")
            os.makedirs(run_save_dir, exist_ok=True)

            scanned_rows = []
            detected_silhouette_key = None
            detected_probe_key = None

            try:
                for row in run.scan_history():
                    if not detected_silhouette_key:
                        detected_silhouette_key = next((k for k in row.keys() if silhouette_keyword in k.lower()), None)
                    if not detected_probe_key:
                        detected_probe_key = next((k for k in row.keys() if probe_keyword in k.lower()), None)

                    has_data = (detected_silhouette_key and row.get(detected_silhouette_key) is not None) or \
                               (detected_probe_key and row.get(detected_probe_key) is not None)

                    if has_data:
                        step_val = row.get("_step", row.get("step", 0))
                        scanned_rows.append({
                            "seed": seed_val,
                            "_step": step_val,
                            "eval/silhouette_score": row.get(detected_silhouette_key),
                            "eval/linear_probe_score": row.get(detected_probe_key),
                        })

                # Save to CSV
                csv_out_path = os.path.join(run_save_dir, f"seed_{seed_val}_metrics.csv")
                pd.DataFrame(scanned_rows).to_csv(csv_out_path, index=False)
                print(f"      📊 Saved to: {os.path.basename(csv_out_path)}")

            except Exception as e:
                print(f"      ❌ Error processing run {run.name}: {e}")

    print(f"\n🎉 Download complete. Data saved in: {base_save_dir}")

if __name__ == "__main__":
    ENTITY = "annika-moeller24-eindhoven-university-of-technology"
    # List the specific projects here
    TARGET_PROJECTS = ["CCDT_Front_Architecture_cw03", "CCDT_Front_Architecture_cw01"]
    LOCAL_SAVE_DIR = "/Users/annikamollerchandiramani/Documents/uni/OSRL_continued/examples/eval/cluster_eval_sweeps"

    download_numerical_thesis_cache(
        entity=ENTITY,
        project_names=TARGET_PROJECTS,
        base_save_dir=LOCAL_SAVE_DIR,
    )