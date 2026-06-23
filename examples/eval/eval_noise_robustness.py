#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import gymnasium as gym
import yaml
import glob

# --- PATH SETUP ---
PROJECT_ROOT = "/home/20234949/thesis/OSRL_continued"
sys.path.insert(0, PROJECT_ROOT)

import bullet_safety_gym  # noqa
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.common.exp_util import load_config_and_model
from osrl.algorithms.cdt import CDT
from osrl.algorithms.ccdt import ContrastiveCDTBack

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- 1. THE NOISE WRAPPER ---
class GaussianNoiseObservationWrapper(gym.ObservationWrapper):
    """Injects Gaussian noise into the state observations to simulate sensor failure."""
    def __init__(self, env, noise_scale=0.0):
        super().__init__(env)
        self.noise_scale = noise_scale

    def observation(self, obs):
        if self.noise_scale > 0.0:
            noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=obs.shape)
            return (obs + noise).astype(np.float32)
        return obs

# --- 2. MODEL LOADING ---
def resolve_config_path(env_name, model_type):
    if model_type == "vanilla":
        pattern = os.path.join(PROJECT_ROOT, f"models/cdt//Vanilla_CDT_Offline{env_name}-v0/**/config.yaml")
    else:
        # Update this path to point to your new threshold/distance models once they finish!
        pattern = os.path.join(PROJECT_ROOT, f"models/ccdt_arch_a/Back_Offline{env_name}-v0_3Buckets_cw03/**/config.yaml")

    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"❌ Could not find config for {model_type}")
    return matches[0]

def load_agent(config_path, model_type):
    exp_dir = os.path.dirname(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.unsafe_load(f)

    try:
        _, model_weights = load_config_and_model(exp_dir, best=True)
    except:
        _, model_weights = load_config_and_model(exp_dir, best=False)

    base_env = gym.make(cfg["task"])
    env = wrap_env(env=base_env, reward_scale=cfg["reward_scale"])
    
    if model_type == "vanilla":
        model = CDT(
            state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
            max_action=env.action_space.high[0], embedding_dim=cfg["embedding_dim"],
            seq_len=cfg["seq_len"], episode_len=cfg["episode_len"], num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"], time_emb=cfg["time_emb"], use_rew=cfg["use_rew"],
            use_cost=cfg["use_cost"], cost_transform=cfg["cost_transform"],
            target_entropy=-env.action_space.shape[0], stochastic=cfg.get("stochastic", True)
        )
    else:
        model = ContrastiveCDTBack(
            state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
            max_action=env.action_space.high[0], embedding_dim=cfg["embedding_dim"],
            contrastive_dim=cfg.get("contrastive_dim", 64), seq_len=cfg["seq_len"],
            episode_len=cfg["episode_len"], num_layers=cfg["num_layers"], num_heads=cfg["num_heads"],
            use_rew=cfg["use_rew"], use_cost=cfg["use_cost"], cost_transform=cfg["cost_transform"],
            stochastic=cfg.get("stochastic", True)
        )

    state_dict = model_weights.get("model_state", model_weights.get("model", model_weights))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    return model, cfg

# --- 3. LIVE ROLLOUT FUNCTION ---
def evaluate_robustness(env_name, noise_levels, num_episodes=5):
    results = []
    models_to_test = ["vanilla", "ccdt"]
    
    for model_type in models_to_test:
        print(f"\n🧠 Loading {model_type.upper()}...")
        config_path = resolve_config_path(env_name, model_type)
        model, cfg = load_agent(config_path, model_type)
        
        target_ret = cfg.get("target_returns", [[500, 10]])[0][0] * cfg["reward_scale"]
        target_cost = cfg.get("target_returns", [[500, 10]])[0][1] * cfg.get("cost_scale", 1.0)
        
        for noise in noise_levels:
            print(f"  🌪️ Testing Noise Level: {noise}...")
            
            # Create fresh environment wrapped with noise
            base_env = gym.make(f"Offline{env_name}-v0")
            env = wrap_env(env=base_env, reward_scale=cfg["reward_scale"])
            env = GaussianNoiseObservationWrapper(env, noise_scale=noise)
            
            total_costs = []
            
            for ep in range(num_episodes):
                state, _ = env.reset()
                ep_cost = 0.0
                
                # Initialize transformer context queues
                states = torch.zeros((1, cfg["seq_len"], env.observation_space.shape[0]), device=DEVICE)
                actions = torch.zeros((1, cfg["seq_len"], env.action_space.shape[0]), device=DEVICE)
                rewards = torch.zeros((1, cfg["seq_len"]), device=DEVICE)
                costs = torch.zeros((1, cfg["seq_len"]), device=DEVICE)
                time_steps = torch.zeros((1, cfg["seq_len"]), dtype=torch.long, device=DEVICE)
                
                ep_ret, ep_cost_to_go = target_ret, target_cost
                
                for t in range(cfg["episode_len"]):
                    states[0, -1] = torch.tensor(state, device=DEVICE)
                    rewards[0, -1] = ep_ret
                    costs[0, -1] = ep_cost_to_go
                    time_steps[0, -1] = t

                    with torch.no_grad():
                        if hasattr(model, "get_action"):
                            try:
                                action = model.get_action(states, actions, rewards, costs, time_steps)
                            except TypeError:
                                action = model.get_action(states, actions, rewards, costs)
                        else:
                            action_preds, _, _ = model(states, actions, rewards, costs, time_steps)
                            
                            # Handle Stochastic Policies (Extract the mean action)
                            if hasattr(action_preds, 'mean'):
                                action = action_preds.mean[:, -1]
                            else:
                                # Handle Deterministic Policies (Standard tensor)
                                action = action_preds[:, -1]

                    action = action.squeeze().cpu().numpy()
                    next_state, reward, terminated, truncated, info = env.step(action)
                    
                    cost = info.get("cost", 0.0)
                    ep_cost += cost
                    
                    # Update context window
                    states = torch.cat([states[:, 1:], torch.zeros((1, 1, states.shape[-1]), device=DEVICE)], dim=1)
                    actions[0, -1] = torch.tensor(action, device=DEVICE)
                    actions = torch.cat([actions[:, 1:], torch.zeros((1, 1, actions.shape[-1]), device=DEVICE)], dim=1)
                    rewards = torch.cat([rewards[:, 1:], torch.zeros((1, 1), device=DEVICE)], dim=1)
                    costs = torch.cat([costs[:, 1:], torch.zeros((1, 1), device=DEVICE)], dim=1)
                    time_steps = torch.cat([time_steps[:, 1:], torch.zeros((1, 1), dtype=torch.long, device=DEVICE)], dim=1)
                    
                    ep_ret -= reward * cfg["reward_scale"]
                    ep_cost_to_go -= cost * cfg.get("cost_scale", 1.0)
                    state = next_state
                    
                    if terminated or truncated:
                        break
                        
                total_costs.append(ep_cost)
            
            avg_cost = np.mean(total_costs)
            std_cost = np.std(total_costs)
            results.append({"Model": model_type.upper(), "Noise": noise, "Avg_Cost": avg_cost, "Std_Cost": std_cost})
            
    return pd.DataFrame(results)

# --- 4. PLOTTING ---
def plot_robustness(df, env_name):
    plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"VANILLA": "#E63946", "CCDT": "#2A9D8F"}
    
    for model in df["Model"].unique():
        subset = df[df["Model"] == model]
        ax.plot(subset["Noise"], subset["Avg_Cost"], marker='o', linewidth=2.5, label=model, color=colors[model])
        ax.fill_between(subset["Noise"], subset["Avg_Cost"] - subset["Std_Cost"], subset["Avg_Cost"] + subset["Std_Cost"], alpha=0.15, color=colors[model])

    ax.axhline(y=10.0, color='black', linestyle='--', alpha=0.5, label="Cost Limit")
    ax.set_xlabel("Sensor Noise (Gaussian Std Dev)", fontweight="bold")
    ax.set_ylabel("Average Trajectory Cost", fontweight="bold")
    ax.set_title(f"Out-of-Distribution Robustness\n{env_name}", pad=15, fontweight="bold")
    ax.legend(frameon=False)
    
    os.makedirs(os.path.join(PROJECT_ROOT, "examples/eval/eval_suite/plots"), exist_ok=True)
    out_path = os.path.join(PROJECT_ROOT, f"examples/eval/eval_suite/plots/robustness_{env_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Plot saved to {out_path}")

if __name__ == "__main__":
    # List all target environments
    target_envs = ["AntRun", "CarCircle", "DroneRun"]
    noise_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0] # How heavily to corrupt the states
    
    for target_env in target_envs:
        print("\n" + "="*50)
        print(f"🚀 Booting OOD Robustness Evaluation for {target_env}...")
        print("="*50)
        
        # Run the evaluation for the current environment
        df_results = evaluate_robustness(target_env, noise_range, num_episodes=10)
        
        print(f"\n📊 Raw Results for {target_env}:")
        print(df_results.to_string(index=False))
        
        # Plot and save
        plot_robustness(df_results, target_env)

    print("\n🏁 All OOD Robustness evaluations complete! Check your plots directory.")