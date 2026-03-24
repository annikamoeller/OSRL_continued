import os
import glob
import pandas as pd
import torch
import traceback
import gymnasium as gym
import sys

# --- PATH SETUP ---
sys.path.insert(0, "/home/20234949/thesis/OSRL_continued")

import bullet_safety_gym  # noqa
import dsrl
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.common.exp_util import load_config_and_model, seed_all
from osrl.algorithms.ccdt import ContrastiveCDT, ContrastiveCDTTrainer

# --- CONFIGURATION ---
LOG_ROOT = "/home/20234949/thesis/logs"
OUTPUT_CSV = "raw_eval_collection.csv"

# Direct sweep from 0 to 80 in increments of 10
TARGET_COST_SWEEP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
NUM_EPISODES = 20 
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def collect_raw_eval_data():
    results = []
    search_pattern = os.path.join(LOG_ROOT, "**", "config.yaml")
    config_files = glob.glob(search_pattern, recursive=True)
    
    print(f"🔍 Found {len(config_files)} experiments. Starting raw collection...")

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

            # 3. Initialize Model 
            model = ContrastiveCDT(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                max_action=env.action_space.high[0],
                embedding_dim=cfg["embedding_dim"],
                contrastive_dim=64, # Fixed to match your checkpoints
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
            
            # 4. Initialize Trainer
            trainer = ContrastiveCDTTrainer(
                model, 
                env, 
                cost_boundaries=[10.0, 20.0, 30.0], # Dummy boundaries to prevent NoneType crash
                device=DEVICE,
                reward_scale=cfg["reward_scale"],
                cost_scale=cfg["cost_scale"]
            )
            
            # 5. Calculate Target Prompts
            # Fetch max_reward from config (fallback to 1000.0) and take 80%
            env_max_reward = cfg.get("max_reward", 1000.0)
            target_reward = 0.8 * env_max_reward
            
            # 6. The Evaluation Sweep
            for target_cost in TARGET_COST_SWEEP:
                print(f"  🚀 Eval | Target Cost: {target_cost} | Target Reward: {target_reward}")
                
                # Multiply by scales before passing to the model
                raw_eval_ret, raw_eval_cost, ep_length = trainer.evaluate(
                    num_rollouts=NUM_EPISODES, 
                    target_return=target_reward * cfg["reward_scale"],
                    target_cost=target_cost * cfg["cost_scale"]
                )
                
                results.append({
                    "Task": cfg["task"].replace("Offline", "").replace("-v0", ""),
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

    # 7. Save the DataFrame
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Collection complete! Raw data saved to {OUTPUT_CSV}")
    else:
        print("\n⚠️ No data collected.")

    return df

if __name__ == "__main__":
    collect_raw_eval_data()