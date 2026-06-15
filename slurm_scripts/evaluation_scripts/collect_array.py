#!/usr/bin/env python3
import os
import glob
import argparse
import sys
import pandas as pd
import traceback

# Ensure repository root is visible to the cluster node
sys.path.insert(0, "/home/20234949/thesis/OSRL_continued")

# Import the modular engines from your newly unified suite
from examples.eval.eval_suite.eval_suite import (
    load_ccdt_model,
    load_vanilla_model,
    load_dataset_stats,
    TARGET_COST_SWEEP,
    TARGET_REWARD_MULTIPLIERS,
    NUM_EPISODES,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_filter", type=str, required=True)
    parser.add_argument("--array_idx", type=int, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["ccdt", "vanilla"], required=True)
    parser.add_argument("--eval_mode", type=str, choices=["cost", "pareto"], required=True)
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    output_csv = os.path.join(args.run_dir, f"part_{args.array_idx}.csv")

    # Route to the correct tracking directory
    if args.model_type == "vanilla":
        LOG_ROOT = "/home/20234949/thesis/OSRL_continued/output_cdt"
    else:
        LOG_ROOT = "/home/20234949/thesis/OSRL_continued/thesis_final_models"

    search_pattern = os.path.join(LOG_ROOT, args.log_filter, "**", "config.yaml")
    config_files = sorted(glob.glob(search_pattern, recursive=True))

    if not config_files:
        print(f"❌ No models found matching filter {args.log_filter} under path: {LOG_ROOT}")
        sys.exit(1)

    if args.array_idx >= len(config_files):
        print(f"⏩ Array index {args.array_idx} out of bounds. Exiting cleanly.")
        sys.exit(0)

    target_config = config_files[args.array_idx]

    print(f"📌 [Array Worker {args.array_idx}] Isolating model run: {os.path.basename(os.path.dirname(target_config))}")
    print(f"🛠️ Mode: {args.model_type.upper()} | Eval: {args.eval_mode.upper()}")

    try:
        # Load the right model architecture
        if args.model_type == "vanilla":
            trainer, cfg, arch, buckets, cw, task_name = load_vanilla_model(target_config)
            variant_name = "Vanilla Baseline"
        else:
            trainer, cfg, arch, buckets, cw, task_name = load_ccdt_model(target_config)
            variant_name = f"{arch}-{buckets}B"

        stats_lookup = load_dataset_stats()
        match = next((k for k in stats_lookup.keys() if task_name in k), None)
        dataset_max_reward = stats_lookup[match]["Return_Max"] if match else 1000.0

        # Define reward matrix based on evaluation mode
        reward_targets = (
            [dataset_max_reward]
            if args.eval_mode == "cost"
            else [dataset_max_reward * m for m in TARGET_REWARD_MULTIPLIERS]
        )

        results = []
        for t_rew in reward_targets:
            for t_cost in TARGET_COST_SWEEP:
                print(f"  🚀 Target Cost: {t_cost} | Target Reward: {t_rew:.1f}")

                raw_ret, raw_cost, ep_len = trainer.evaluate(
                    num_rollouts=NUM_EPISODES,
                    target_return=t_rew * cfg["reward_scale"],
                    target_cost=t_cost * cfg["cost_scale"],
                )

                results.append(
                    {
                        "Task": task_name,
                        "Seed": cfg["seed"],
                        "Architecture": arch,
                        "Buckets": buckets,
                        "Contrastive_Weight": cw,
                        "Variant": variant_name,
                        "Target_Cost": t_cost,
                        "Target_Reward": t_rew,
                        "Raw_Eval_Cost": raw_cost,
                        "Raw_Eval_Reward": raw_ret,
                        "Avg_Episode_Length": ep_len,
                    }
                )

        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"✅ Worker {args.array_idx} successfully generated {os.path.basename(output_csv)}")

    except Exception as e:
        print(f"❌ Worker {args.array_idx} pipeline execution crash.")
        traceback.print_exc()
        sys.exit(1)
