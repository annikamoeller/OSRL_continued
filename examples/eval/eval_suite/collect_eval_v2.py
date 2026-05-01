import os
import glob
import pandas as pd
import torch
import traceback
import gymnasium as gym
import sys
import datetime
import argparse

# --- PATH SETUP ---
sys.path.insert(0, "/home/20234949/thesis/OSRL_continued")

import bullet_safety_gym  # noqa
import dsrl
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.common.exp_util import load_config_and_model, seed_all
from osrl.algorithms.ccdt import ContrastiveCDTFront, ContrastiveCDTBack, ContrastiveCDTTrainer

# --- DYNAMIC FOLDER & STATS SETUP ---
LOG_ROOT = "/home/20234949/thesis/OSRL_continued/logs"
BASE_EVAL_DIR = "examples/eval/eval_suite"
STATS_CSV = "/home/20234949/thesis/OSRL_continued/dataset_analysis/master_dataset_stats.csv"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
RUN_DIR = os.path.join(BASE_EVAL_DIR, f"eval_{timestamp}")
os.makedirs(RUN_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(RUN_DIR, "raw_data.csv")

TARGET_COST_SWEEP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
NUM_EPISODES = 20 
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def collect_raw_eval_data(log_filter):
    results = [] # We keep this just to return at the end if needed
    
    if not os.path.exists(STATS_CSV):
        raise FileNotFoundError(f"❌ Missing {STATS_CSV}. Check the absolute path.")
    
    print(f"📊 Loading Ground Truth Dataset Stats from {STATS_CSV}...")
    stats_df = pd.read_csv(STATS_CSV)
    stats_lookup = stats_df.set_index("Task").to_dict('index')
    
    # --- DYNAMIC SEARCH PATTERN ---
    # Uses the log_filter argument
    search_pattern = os.path.join(LOG_ROOT, log_filter, "**", "config.yaml")
    config_files = glob.glob(search_pattern, recursive=True)
    
    if not config_files:
        print(f"❌ No config files found matching pattern: {search_pattern}")
        return

    print(f"🔍 Found {len(config_files)} experiments. Starting raw collection...")

    columns = ["Task", "Seed", "Architecture", "Buckets", "Variant", 
               "Target_Cost", "Target_Reward", "Raw_Eval_Cost", "Raw_Eval_Reward", "Avg_Episode_Length"]
    pd.DataFrame(columns=columns).to_csv(OUTPUT_CSV, index=False)

    for config_path in config_files:
        exp_dir = os.path.dirname(config_path)
        print(f"\n📦 Loading: {exp_dir}")
        
        try:
            # 1. Load Config & Model Weights
            try:
                cfg, model_weights = load_config_and_model(exp_dir, best=False)
            except:
                cfg, model_weights = load_config_and_model(exp_dir, best=True)

            seed_all(cfg["seed"])
        
            # 2. Environment Setup
            base_env = gym.make(cfg["task"])
            env = wrap_env(env=base_env, reward_scale=cfg["reward_scale"])
            env = OfflineEnvWrapper(env)

            # 3. Dynamic Architecture Selection
            project_name = cfg.get("project", "")
            encoder_type = cfg.get("encoder_type", "front").lower() 
            is_back_encoder = "back" in project_name.lower() or encoder_type == "back"
            ModelClass = ContrastiveCDTBack if is_back_encoder else ContrastiveCDTFront
            
            arch_label = "Back" if is_back_encoder else "Front"
            print(f"  🧠 Detected Architecture: {arch_label}-Encoder")

            # 4. Initialize Model 
            model = ModelClass(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                max_action=env.action_space.high[0],
                embedding_dim=cfg["embedding_dim"],
                contrastive_dim=cfg.get("contrastive_dim", 64), 
                seq_len=cfg["seq_len"],
                episode_len=cfg["episode_len"],
                num_layers=cfg["num_layers"],
                num_heads=cfg["num_heads"],
                use_rew=cfg["use_rew"],
                use_cost=cfg["use_cost"],
                cost_transform=cfg["cost_transform"],
                stochastic=cfg.get("stochastic", False),
            )
            
            state_dict = model_weights.get("model_state", model_weights.get("model", model_weights))
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            
            # 5. Initialize Trainer
            trainer = ContrastiveCDTTrainer(
                model, env, cost_boundaries=None, device=DEVICE,
                reward_scale=cfg["reward_scale"], cost_scale=cfg["cost_scale"]
            )
            
            # 6. Calculate Target Prompts
            clean_task_name = cfg["task"].replace("Offline", "").replace("-v0", "")
            match = next((k for k in stats_lookup.keys() if clean_task_name in k), None)
            dataset_max_reward = stats_lookup[match]["Return_Max"] if match else cfg.get("max_reward", 1000.0)
                
            target_reward = 1.0 * dataset_max_reward
            num_buckets = cfg.get('num_buckets', 1)
            
            # 7. The Evaluation Sweep
            for target_cost in TARGET_COST_SWEEP:
                print(f"  🚀 Eval | Target Cost: {target_cost} | Target Reward: {target_reward:.1f}")
                
                raw_eval_ret, raw_eval_cost, ep_length = trainer.evaluate(
                    num_rollouts=NUM_EPISODES, 
                    target_return=target_reward * cfg["reward_scale"],
                    target_cost=target_cost * cfg["cost_scale"]
                )
                
                # Create the data dictionary
                row_data = {
                    "Task": clean_task_name,
                    "Seed": cfg["seed"],
                    "Architecture": arch_label,               
                    "Buckets": num_buckets,                   
                    "Variant": f"{arch_label}-{num_buckets}B",
                    "Target_Cost": target_cost,
                    "Target_Reward": target_reward,
                    "Raw_Eval_Cost": raw_eval_cost,
                    "Raw_Eval_Reward": raw_eval_ret,
                    "Avg_Episode_Length": ep_length
                }
                
                results.append(row_data)
                
                # --- THE FIX 2: INCREMENTAL SAVE ---
                # Convert this single row to a DataFrame and append it to the CSV
                pd.DataFrame([row_data]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
                
        except Exception as e:
            print(f"❌ Error processing {exp_dir}:")
            traceback.print_exc()

    print(f"\n✅ Collection complete! Data safely stored in: {RUN_DIR}")
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_filter", type=str, default="Bucket_Sweep_cw04_*", 
                        help="Folder pattern in logs/ to search for")
    args = parser.parse_args()
    
    # This passes the argument from your bash script into the function
    collect_raw_eval_data(args.log_filter)