#!/usr/bin/env python3
import os
import wandb


def download_only_run_images(entity: str, unified_project: str, base_save_dir: str):
    """
    Surgically downloads physical logged image assets from the backend file registry,
    completely bypassing slow row-by-row history logging streams.
    """
    print(f"🚀 Initializing Targeted Image Artifact Downloader from: {entity}/{unified_project}")
    api = wandb.Api()

    try:
        runs = api.runs(f"{entity}/{unified_project}", filters={"state": "finished"})
    except Exception as e:
        print(f"❌ Failed to access W&B project: {e}")
        return

    total_runs = len(runs)
    print(f"✅ Found {total_runs} total runs to scan for cluster images.\n")

    for idx, run in enumerate(runs):
        cfg = run.config
        env = cfg.get("task")
        buckets = cfg.get("num_buckets")
        seed_val = cfg.get("seed")

        contrastive_weight = cfg.get("contrastive_weight", 0.0)
        if contrastive_weight == 0.0 and "args" in cfg and isinstance(cfg["args"], dict):
            contrastive_weight = cfg["args"].get("contrastive_weight", 0.0)

        try:
            buckets = int(float(buckets)) if buckets is not None else 1
            cw_val = float(contrastive_weight)
            seed_val = int(float(seed_val)) if seed_val is not None else 0
        except (ValueError, TypeError):
            continue

        if not env:
            continue

        # Reconstruct standard directory hierarchy
        project_name = cfg.get("project", "")
        encoder_type = str(cfg.get("encoder_type", "front")).lower()
        is_back = "back" in project_name.lower() or encoder_type == "back" or "back" in run.name.lower()
        arch_str = "Back" if is_back else "Front"
        cw_str = f"{cw_val:.2f}".replace("0.", "cw")
        legacy_project_dirname = f"CCDT_Arch_{arch_str}_{cw_str}"

        # Group target images cleanly inside your local workspace structure
        image_save_dir = os.path.join(
            base_save_dir, env, legacy_project_dirname, f"{buckets}B", "images", f"seed_{seed_val}"
        )

        try:
            # Grab references directly from the run's uploaded files manifest (extremely fast)
            image_files = [f for f in run.files() if f.name.endswith((".png", ".jpg", ".jpeg"))]

            if image_files:
                os.makedirs(image_save_dir, exist_ok=True)
                print(
                    f"🖼️  [{idx+1}/{total_runs}] Syncing {len(image_files)} cluster charts for Seed {seed_val} -> {env} ({arch_str} {cw_str})"
                )

                for f in image_files:
                    # Clean up file path structure from wandb path labels
                    clean_filename = os.path.basename(f.name)
                    # Skip basic default system media plots if any
                    if "media" in f.name and not clean_filename.startswith("media"):
                        clean_filename = f.name.replace("/", "_")

                    target_local_path = os.path.join(image_save_dir, clean_filename)

                    # Direct file stream down to your laptop disk space
                    f.download(root=image_save_dir, replace=True)

                    # If wandb nesting left artifacts, cleanly flatten it out
                    nested_artifact = os.path.join(image_save_dir, f.name)
                    if os.path.exists(nested_artifact) and nested_artifact != target_local_path:
                        os.rename(nested_artifact, target_local_path)
            else:
                print(f"⏩ [{idx+1}/{total_runs}] Skipping Seed {seed_val} (No raw charts found).")

        except Exception as e:
            print(f"   ❌ Error pulling assets for run {run.id}: {e}")

    # Quick sweep to drop empty tracking remnants left by wandb core download blocks
    for root, dirs, files in os.walk(base_save_dir, topdown=False):
        for name in dirs:
            if name == "media":
                import shutil

                shutil.rmtree(os.path.join(root, name), ignore_errors=True)

    print(f"\n🎉 Image cache pass complete! Clustered visuals dropped to: {base_save_dir}")


if __name__ == "__main__":
    ENTITY = "annika-moeller24-eindhoven-university-of-technology"
    NEW_PROJECT = "CCDT_Final_Thesis_Baselines"
    LOCAL_SAVE_DIR = "/Users/annikamollerchandiramani/Documents/uni/OSRL_continued/examples/eval/cluster_eval_sweeps"

    download_only_run_images(entity=ENTITY, unified_project=NEW_PROJECT, base_save_dir=LOCAL_SAVE_DIR)
