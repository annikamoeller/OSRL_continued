import os
import glob
import pandas as pd
import torch
import traceback
import gymnasium as gym
import sys
import datetime
import argparse
import yaml
import numpy as np

# --- PATH SETUP ---
sys.path.insert(0, "/home/20234949/thesis/OSRL_continued")

import bullet_safety_gym  # noqa
import dsrl
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.common.exp_util import seed_all
from osrl.algorithms.cdt import CDT, CDTTrainer

# --- DYNAMIC FOLDER & STATS SETUP ---
LOG_ROOT = "/home/20234949/thesis/OSRL_continued/output_cdt"
BASE_EVAL_DIR = "examples/eval/eval_suite"
STATS_CSV = "/home/20234949/thesis/OSRL_continued/dataset_analysis/master_dataset_stats.csv"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
RUN_DIR = os.path.join(BASE_EVAL_DIR, f"eval_vanilla_{timestamp}")
os.makedirs(RUN_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(RUN_DIR, "raw_vanilla_data.csv")

TARGET_COST_SWEEP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
NUM_EPISODES = 20 
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def collect_vanilla_eval_data(log_filter):
    global OUTPUT_CSV  
    results = [] 
    
    if not os.path.exists(STATS_CSV):
        raise FileNotFoundError(f"❌ Missing {STATS_CSV}. Check the absolute path.")
    
    print(f"📊 Loading Stats from {STATS_CSV}...")
    stats_df = pd.read_csv(STATS_CSV)
    stats_lookup = stats_df.set_index("Task").to_dict('index')
    
    search_pattern = os.path.join(LOG_ROOT, log_filter, "**", "config.yaml")
    config_files = glob.glob(search_pattern, recursive=True)
    
    if not config_files:
        print(f"❌ No Vanilla baseline configs found matching pattern: {search_pattern}")
        return

    print(f"🔍 Found {len(config_files)} vanilla experiments. Starting direct rollout sweep...")

    for config_path in config_files:
        exp_dir = os.path.dirname(config_path)
        print(f"\n📦 Loading Vanilla Baseline: {exp_dir}")
        
        try:
            def construct_yaml_tuple(loader, node):
                return tuple(loader.construct_sequence(node))
            yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', construct_yaml_tuple)

            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)

            # Resolve checkpoint weight targets
            checkpoint_dir = os.path.join(exp_dir, "checkpoint")
            model_path = os.path.join(checkpoint_dir, "model_best.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join(checkpoint_dir, "model.pt")
                
            model_weights = torch.load(model_path, map_location=torch.device(DEVICE))
            seed_all(cfg["seed"])
        
            # --- AUTHOR ENVIRONMENT MATCHING LAYOUT ---
            base_env = gym.make(cfg["task"])
            env = wrap_env(env=base_env, reward_scale=cfg["reward_scale"])
            env = OfflineEnvWrapper(env)
            
            # Match the author's explicit environment cost boundary pinning
            if "cost_limit" in cfg:
                env.set_target_cost(cfg["cost_limit"])

            print("  🧠 Building Native Linear Baseline CDT Topology...")
            target_entropy = -env.action_space.shape[0]
            model = CDT(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                max_action=env.action_space.high[0],
                embedding_dim=cfg["embedding_dim"],
                seq_len=cfg["seq_len"],
                episode_len=cfg["episode_len"],
                num_layers=cfg["num_layers"],
                num_heads=cfg["num_heads"],
                attention_dropout=cfg.get("attention_dropout", 0.0),
                residual_dropout=cfg.get("residual_dropout", 0.0),
                embedding_dropout=cfg.get("embedding_dropout", 0.0),
                time_emb=cfg["time_emb"],
                use_rew=cfg["use_rew"],
                use_cost=cfg["use_cost"],
                cost_transform=cfg["cost_transform"],
                add_cost_feat=cfg.get("add_cost_feat", False),
                mul_cost_feat=cfg.get("mul_cost_feat", False),
                cat_cost_feat=cfg.get("cat_cost_feat", False),
                action_head_layers=cfg.get("action_head_layers", 1),
                cost_prefix=cfg.get("cost_prefix", False),
                stochastic=cfg.get("stochastic", False),
                init_temperature=cfg.get("init_temperature", 0.1),
                target_entropy=target_entropy,
            )
            
            state_dict = model_weights.get("model_state", model_weights.get("model", model_weights))
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            
            # --- AUTHOR TRAINER INSTANTIATION ---
            # Explicitly pass the cost_reverse hyperparameter out of the configuration map
            trainer = CDTTrainer(
                model=model,
                env=env,
                reward_scale=cfg["reward_scale"],
                cost_scale=cfg["cost_scale"],
                cost_reverse=cfg.get("cost_reverse", False),
                device=DEVICE
            )
            
            clean_task_name = cfg["task"].replace("Offline", "").replace("-v0", "")
            match = next((k for k in stats_lookup.keys() if clean_task_name in k), None)
            dataset_max_reward = stats_lookup[match]["Return_Max"] if match else cfg.get("max_reward", 1000.0)
            target_reward = 1.0 * dataset_max_reward
            
            for target_cost in TARGET_COST_SWEEP:
                print(f"    🚀 Direct Rollout | Target Cost: {target_cost} | Target Reward: {target_reward:.1f}")
                
                # Use the native trainer evaluation engine directly
                raw_eval_ret, raw_eval_cost, ep_length = trainer.evaluate(
                    num_rollouts=NUM_EPISODES,
                    target_return=target_reward * cfg["reward_scale"],
                    target_cost=target_cost * cfg["cost_scale"]
                )
                
                row_data = {
                    "Task": clean_task_name,
                    "Seed": cfg["seed"],
                    "Architecture": "Vanilla",               
                    "Buckets": "Baseline",
                    "Contrastive_Weight": 0.0,                  
                    "Variant": "Vanilla Baseline",
                    "Target_Cost": target_cost,
                    "Target_Reward": target_reward,
                    "Raw_Eval_Cost": raw_eval_cost,
                    "Raw_Eval_Reward": raw_eval_ret,
                    "Avg_Episode_Length": ep_length
                }
                
                results.append(row_data)
                write_header = not os.path.exists(OUTPUT_CSV)
                pd.DataFrame([row_data]).to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)
                
        except Exception as e:
            print(f"❌ Error evaluating Vanilla run {exp_dir}:")
            traceback.print_exc()

    print(f"\n✅ Vanilla baseline evaluation finished! Target file: {OUTPUT_CSV}")
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_filter", type=str, default="Vanilla_CDT*", 
                        help="Folder prefix patterns matching your baseline runs")
    args = parser.parse_args()
    
    collect_vanilla_eval_data(args.log_filter)