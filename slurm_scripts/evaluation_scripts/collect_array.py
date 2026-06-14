#!/usr/bin/env python3
import os
import glob
import argparse
import sys

# Ensure repository root is visible to the cluster node
sys.path.insert(0, "/home/20234949/thesis/OSRL_continued")

# Target your actual script module name
import examples.eval.eval_suite.collect_eval as base_script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_filter", type=str, required=True)
    parser.add_argument("--array_idx", type=int, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.run_dir, exist_ok=True)
    
    # Force the unique parallel worker slice naming format so files don't overwrite
    base_script.OUTPUT_CSV = os.path.join(args.run_dir, f"part_{args.array_idx}.csv")
    
    # --- AUTOMATIC ROOT ROUTING ---
    if "Vanilla" in args.log_filter:
        LOG_ROOT = "/home/20234949/thesis/OSRL_continued/output_cdt"
    elif "cw" in args.log_filter:
        # 🌟 FIXED: Point contrastive weight runs directly to your clean workspace
        LOG_ROOT = "/home/20234949/thesis/OSRL_continued/thesis_final_models"
    else:
        LOG_ROOT = "/home/20234949/thesis/OSRL_continued/output"
        
    base_script.LOG_ROOT = LOG_ROOT
    
    search_pattern = os.path.join(LOG_ROOT, args.log_filter, "**", "config.yaml")
    config_files = sorted(glob.glob(search_pattern, recursive=True))
    
    if not config_files:
        print(f"❌ No models found matching filter {args.log_filter} under path: {LOG_ROOT}")
        sys.exit(1)
        
    if args.array_idx >= len(config_files):
        print(f"⏩ Array index {args.array_idx} out of bounds ({len(config_files)} files total). Exiting cleanly.")
        sys.exit(0)
        
    target_config = config_files[args.array_idx]
    target_dir = os.path.dirname(target_config)
    
    print(f"📌 [Array Worker {args.array_idx}] isolating model run: {os.path.basename(target_dir)}")
    print(f"📂 Config target: {target_config}")
    
    # --- SAFE INTERCEPTOR MONKEY-PATCH ---
    import glob as global_glob
    real_glob = global_glob.glob
    
    def scoped_glob_patch(pattern, recursive=True):
        # Only intercept if the base script is specifically hunting down its config file list
        if "config.yaml" in pattern:
            return [target_config]
        # Pass through to the real glob for weight-loading (*.pt) or anything else!
        return real_glob(pattern, recursive=recursive)
        
    global_glob.glob = scoped_glob_patch
    
    # Execute evaluation pipeline safely inside your single isolated run
    try:
        base_script.collect_ccdt_eval_data("")
        print(f"✅ Worker {args.array_idx} successfully generated part_{args.array_idx}.csv")
    except Exception as e:
        print(f"❌ Worker {args.array_idx} pipeline execution crash.")
        import traceback
        traceback.print_exc()
        sys.exit(1)