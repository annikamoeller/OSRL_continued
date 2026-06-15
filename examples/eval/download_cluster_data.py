#!/usr/bin/env python3
import os
import pandas as pd
import wandb


def download_numerical_thesis_cache(
    entity: str,
    unified_project: str,
    base_save_dir: str,
    silhouette_keyword: str = "silhouette",
    probe_keyword: str = "probe",
):
    """
    Speed-optimized pipeline: Downloads ONLY numerical evaluation metrics (silhouette and linear probe scores)
    for ALL finished runs inside the project, completely skipping rich media binary files.
    """
    print(f"🚀 Initializing Fast Numerical Asset Download Pipeline from: {entity}/{unified_project}")
    api = wandb.Api()

    try:
        print("📡 Querying W&B server for all finished runs...")
        runs = api.runs(f"{entity}/{unified_project}", filters={"state": "finished"})
    except Exception as e:
        print(f"❌ Failed to access W&B project: {e}")
        return

    total_runs = len(runs)
    print(f"✅ Found {total_runs} total verified baseline runs to process.\n")

    for idx, run in enumerate(runs):
        cfg = run.config
        env = cfg.get("task")
        buckets = cfg.get("num_buckets")
        seed_val = cfg.get("seed")

        # Extract contrastive weight dynamically
        contrastive_weight = cfg.get("contrastive_weight", 0.0)
        if contrastive_weight == 0.0 and "args" in cfg and isinstance(cfg["args"], dict):
            contrastive_weight = cfg["args"].get("contrastive_weight", 0.0)

        try:
            buckets = int(float(buckets)) if buckets is not None else 1
            cw_val = float(contrastive_weight)
            seed_val = int(float(seed_val)) if seed_val is not None else 0
        except ValueError:
            continue

        if not env:
            continue

        # Determine architecture encoder label
        project_name = cfg.get("project", "")
        encoder_type = str(cfg.get("encoder_type", "front")).lower()
        is_back = "back" in project_name.lower() or encoder_type == "back" or "back" in run.name.lower()
        arch_str = "Back" if is_back else "Front"

        # Formulate standard weight suffix string (e.g., 0.03 -> "cw03")
        cw_str = f"{cw_val:.2f}".replace("0.", "cw")
        legacy_project_dirname = f"CCDT_Arch_{arch_str}_{cw_str}"

        print(
            f"📥 [{idx+1}/{total_runs}] Syncing metrics: {env} | {arch_str} ({cw_str}) | {buckets}B | Seed {seed_val}"
        )

        # Establish path structure
        run_save_dir = os.path.join(base_save_dir, env, legacy_project_dirname, f"{buckets}B")
        os.makedirs(run_save_dir, exist_ok=True)

        scanned_rows = []
        detected_silhouette_key = None
        detected_probe_key = None

        try:
            # Stream tabular validation histories only
            for row in run.scan_history():
                if not detected_silhouette_key:
                    detected_silhouette_key = next((k for k in row.keys() if silhouette_keyword in k.lower()), None)

                if not detected_probe_key:
                    detected_probe_key = next((k for k in row.keys() if probe_keyword in k.lower()), None)

                has_silhouette = detected_silhouette_key and row.get(detected_silhouette_key) is not None
                has_probe = detected_probe_key and row.get(detected_probe_key) is not None

                if has_silhouette or has_probe:
                    step_val = row.get("_step") if "_step" in row else row.get("step", 0)

                    scanned_rows.append(
                        {
                            "seed": seed_val,
                            "_step": step_val,
                            "eval/silhouette_score": row.get(detected_silhouette_key) if has_silhouette else None,
                            "eval/linear_probe_score": row.get(detected_probe_key) if has_probe else None,
                        }
                    )

            # Save the clean numerical csv
            csv_out_path = os.path.join(run_save_dir, f"seed_{seed_val}_metrics_history.csv")
            if scanned_rows:
                metrics_df = pd.DataFrame(scanned_rows).sort_values("_step")
                metrics_df.to_csv(csv_out_path, index=False)
                print(f"   📊 Saved numerical table: {os.path.basename(csv_out_path)}")
            else:
                pd.DataFrame(columns=["seed", "_step", "eval/silhouette_score", "eval/linear_probe_score"]).to_csv(
                    csv_out_path, index=False
                )

        except Exception as e:
            print(f"   ❌ Error aggregating run components: {e}")

    print(f"\n🎉 Clean matrix download complete! Fast numerical data dropped to: {base_save_dir}")


if __name__ == "__main__":
    ENTITY = "annika-moeller24-eindhoven-university-of-technology"
    NEW_PROJECT = "CCDT_Final_Thesis_Baselines"
    LOCAL_SAVE_DIR = "/Users/annikamollerchandiramani/Documents/uni/OSRL_continued/examples/eval/cluster_eval_sweeps"

    download_numerical_thesis_cache(
        entity=ENTITY,
        unified_project=NEW_PROJECT,
        base_save_dir=LOCAL_SAVE_DIR,
        silhouette_keyword="silhouette",
        probe_keyword="probe",
    )
