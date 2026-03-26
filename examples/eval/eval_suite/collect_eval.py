import os
import glob
import pandas as pd
import torch
import traceback
import gymnasium as gym
import sys
import datetime

# --- PATH SETUP ---
sys.path.insert(0, "/home/20234949/thesis/OSRL_continued")

import bullet_safety_gym  # noqa
import dsrl
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.common.exp_util import load_config_and_model, seed_all
from osrl.algorithms.ccdt import ContrastiveCDT, ContrastiveCDTTrainer

# --- DYNAMIC FOLDER & STATS SETUP ---
LOG_ROOT = "/home/20234949/thesis/OSRL_continued/logs"
BASE_EVAL_DIR = "examples/eval/eval_suite"
STATS_CSV = "/home/20234949/thesis/OSRL_continued/dataset_analysis/master_dataset_stats.csv"

# Create a new folder named by the current minute
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
RUN_DIR = os.path.join(BASE_EVAL_DIR, f"eval_{timestamp}")
os.makedirs(RUN_DIR, exist_ok=True)

# Standardize the output name inside this specific folder
OUTPUT_CSV = os.path.join(RUN_DIR, "raw_data.csv")

TARGET_COST_SWEEP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
NUM_EPISODES = 20 
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def collect_raw_eval_data():
    results = []
    
    # 1. Load the Ground Truth Dataset Stats
    if not os.path.exists(STATS_CSV):
        raise FileNotFoundError(f"❌ Missing {STATS_CSV}. Check the absolute path.")
    
    print(f"📊 Loading Ground Truth Dataset Stats from {STATS_CSV}...")
    stats_df = pd.read_csv(STATS_CSV)
    stats_lookup = stats_df.set_index("Task").to_dict('index')
    
    search_pattern = os.path.join(LOG_ROOT, "**", "config.yaml")
    config_files = glob.glob(search_pattern, recursive=True)
    
    print(f"🔍 Found {len(config_files)} experiments. Starting raw collection...")

    for config_path in config_files:
        exp_dir = os.path.dirname(config_path)
        print(f"\n📦 Loading: {exp_dir}")
        
        try:
            # 2. Load Config & Model Weights
            try:
                cfg, model_weights = load_config_and_model(exp_dir, best=False)
            except:
                cfg, model_weights = load_config_and_model(exp_dir, best=True)

            seed_all(cfg["seed"])
        
            # 3. Environment Setup
            base_env = gym.make(cfg["task"])
            env = wrap_env(env=base_env, reward_scale=cfg["reward_scale"])
            env = OfflineEnvWrapper(env)

            # 4. Initialize Model 
            model = ContrastiveCDT(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                max_action=env.action_space.high[0],
                embedding_dim=cfg["embedding_dim"],
                contrastive_dim=64, 
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
                model, 
                env, 
                cost_boundaries=[10.0, 20.0, 30.0], # Dummy boundaries to prevent NoneType crash
                device=DEVICE,
                reward_scale=cfg["reward_scale"],
                cost_scale=cfg["cost_scale"]
            )
            
            # 6. Calculate Target Prompts (The Bulletproof Way)
            clean_task_name = cfg["task"].replace("Offline", "").replace("-v0", "")
            match = next((k for k in stats_lookup.keys() if clean_task_name in k), None)
            
            if match:
                dataset_max_reward = stats_lookup[match]["Return_Max"]
                print(f"  🎯 Exact Match Found! Max Dataset Reward for {clean_task_name}: {dataset_max_reward}")
            else:
                print(f"  ⚠️ No exact stats match for {clean_task_name}. Falling back to config.")
                dataset_max_reward = cfg.get("max_reward", 1000.0)
                
            # Set target to exactly 100% of what the model has actually seen
            target_reward = 1.0 * dataset_max_reward
            
            # 7. The Evaluation Sweep
            for target_cost in TARGET_COST_SWEEP:
                print(f"  🚀 Eval | Target Cost: {target_cost} | Target Reward: {target_reward:.1f}")
                
                raw_eval_ret, raw_eval_cost, ep_length = trainer.evaluate(
                    num_rollouts=NUM_EPISODES, 
                    target_return=target_reward * cfg["reward_scale"],
                    target_cost=target_cost * cfg["cost_scale"]
                )
                
                results.append({
                    "Task": clean_task_name,
                    "Seed": cfg["seed"],
                    "Variant": f"CCDT-{cfg.get('num_buckets', 1)}B",
                    "Target_Cost": target_cost,
                    "Target_Reward": target_reward,
                    "Raw_Eval_Cost": raw_eval_cost,
                    "Raw_Eval_Reward": raw_eval_ret,
                    "Avg_Episode_Length": ep_length
                })
                
        except Exception as e:
            print(f"❌ Error processing {exp_dir}:")
            traceback.print_exc()

    # 8. Save the DataFrame
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Collection complete! Data saved to: {RUN_DIR}")
        print(f"👉 To complete the pipeline, run:")
        print(f"   python examples/eval/plot_eval.py {RUN_DIR}")
        print(f"   python examples/eval/table_eval.py {RUN_DIR}")
    else:
        print("\n⚠️ No data collected.")

    return df

if __name__ == "__main__":
    collect_raw_eval_data()