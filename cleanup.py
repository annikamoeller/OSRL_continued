#!/usr/bin/env python3
import os
import shutil
import glob
import yaml
import wandb

def segregate_and_deduplicate_models_preserve_hierarchy(
    entity: str,
    wandb_project: str,
    root_checkpoint_dir: str,
    target_clean_dir: str,
    target_archive_dir: str,
    dry_run: bool = True,
):
    """
    Extracts run names from local config files, cross-references with active W&B runs,
    and moves the entire experiment folder while preserving the parent directory 
    (e.g., Back_OfflineAntRun-v0_2Buckets_cw01) to keep plotting functions functional.
    """
    print(f"📡 Fetching live run names from {entity}/{wandb_project}...")
    api = wandb.Api()

    try:
        active_runs = api.runs(f"{entity}/{wandb_project}")
        valid_run_names = {run.name for run in active_runs if run.name}
    except Exception as e:
        print(f"❌ Failed to reach W&B: {e}")
        return

    print(f"✅ Found {len(valid_run_names)} active target runs on the cloud.")
    print(f"🔍 Deep scanning paths inside: {root_checkpoint_dir}...\n")

    # Track discovered valid runs: { run_name: [(target_move_dir, parent_dirname, modification_time), ...] }
    valid_matches_registry = {}
    archive_pool = [] # List of tuples: (target_move_dir, parent_dirname)
    processed_paths = set()

    config_files = glob.glob(os.path.join(root_checkpoint_dir, "**/config.yaml"), recursive=True)

    for config_path in config_files:
        current_dir = os.path.dirname(config_path)
        if os.path.basename(current_dir) == "checkpoint":
            current_dir = os.path.dirname(current_dir)
            
        parent_dir = os.path.dirname(current_dir)
        if os.path.basename(parent_dir) == os.path.basename(current_dir):
            target_move_dir = parent_dir
        else:
            target_move_dir = current_dir

        if target_move_dir in processed_paths:
            continue
        processed_paths.add(target_move_dir)

        # Extract the crucial parent folder name (e.g., 'Back_OfflineAntRun-v0_2Buckets_cw01')
        # target_move_dir is /.../output/Back_OfflineAntRun-v0_2Buckets_cw01/AntRun_2B_...
        # so os.path.dirname(target_move_dir) gets us the path up to 'output'
        # and os.path.basename of that directory gets the parent name
        parent_dirname = os.path.basename(os.path.dirname(target_move_dir))

        local_run_name = None
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            name_entry = config_data.get("name", config_data.get("exp_name", {}))
            local_run_name = name_entry.get("value") if isinstance(name_entry, dict) else name_entry
        except Exception:
            local_run_name = None

        if not local_run_name:
            local_run_name = os.path.basename(target_move_dir)

        if local_run_name in valid_run_names and local_run_name is not None:
            mod_time = os.path.getmtime(target_move_dir)
            if local_run_name not in valid_matches_registry:
                valid_matches_registry[local_run_name] = []
            valid_matches_registry[local_run_name].append((target_move_dir, parent_dirname, mod_time))
        else:
            archive_pool.append((target_move_dir, parent_dirname))

    # Deduplication pass
    final_keep_pool = [] # List of tuples: (target_move_dir, parent_dirname)
    duplicate_count = 0

    for name, paths_meta in valid_matches_registry.items():
        # Sort by modification time in descending order (newest first)
        sorted_paths = sorted(paths_meta, key=lambda x: x[2], reverse=True)
        
        # Keep the newest complete run
        final_keep_pool.append((sorted_paths[0][0], sorted_paths[0][1]))
        
        # Move older duplicates to the archive pool
        for duplicate_meta in sorted_paths[1:]:
            archive_pool.append((duplicate_meta[0], duplicate_meta[1]))
            duplicate_count += 1

    # Execute file operations with folder creation logic
    kept_count = 0
    archived_count = 0

    print(f"📦 Processing file distribution operations...\n")
    
    for path, parent_folder in final_keep_pool:
        folder_name = os.path.basename(path)
        # Nest inside the kept parent directory name
        destination_dir = os.path.join(target_clean_dir, parent_folder)
        destination_path = os.path.join(destination_dir, folder_name)
        kept_count += 1
        
        if dry_run:
            print(f"  [KEEP] Path preserved structure:\n         {path}\n         ──> {destination_path}\n")
        else:
            if os.path.exists(path):
                os.makedirs(destination_dir, exist_ok=True)
                shutil.move(path, destination_path)

    for path, parent_folder in archive_pool:
        folder_name = os.path.basename(path)
        destination_dir = os.path.join(target_archive_dir, parent_folder)
        destination_path = os.path.join(destination_dir, folder_name)
        archived_count += 1
        
        if not dry_run and os.path.exists(path):
            os.makedirs(destination_dir, exist_ok=True)
            shutil.move(path, destination_path)

    print(f"📊 Reorganization Summary Layout:")
    print(f"   - Clean Active Matrix Trees Isolated: {kept_count}")
    print(f"   - Redundant Duplicate Runs Archived:  {duplicate_count}")
    print(f"   - Total Folders Sent to Archive:      {archived_count}")

    if dry_run:
        print("\n⚠️ This was a DRY RUN. No files were shifted on your HPC account.")
        print("💡 Set `dry_run=False` when you are ready to execute the cleanup.")


if __name__ == "__main__":
    ENTITY = "annika-moeller24-eindhoven-university-of-technology"
    CLEAN_PROJECT = "CCDT_Final_Thesis_Baselines"

    BASE_OUTPUT_PATH = "/home/20234949/thesis/OSRL_continued/output/"
    FINAL_TARGET_DIR = "/home/20234949/thesis/OSRL_continued/thesis_final_models"
    ARCHIVE_TARGET_DIR = "/home/20234949/thesis/OSRL_continued/archive_models"

    segregate_and_deduplicate_models_preserve_hierarchy(
        entity=ENTITY,
        wandb_project=CLEAN_PROJECT,
        root_checkpoint_dir=BASE_OUTPUT_PATH,
        target_clean_dir=FINAL_TARGET_DIR,
        target_archive_dir=ARCHIVE_TARGET_DIR,
        dry_run=False,
    )