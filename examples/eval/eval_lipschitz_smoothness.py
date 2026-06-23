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
import random

# --- PATH SETUP ---
PROJECT_ROOT = "/home/20234949/thesis/OSRL_continued"
sys.path.insert(0, PROJECT_ROOT)

import bullet_safety_gym  # noqa
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.common.exp_util import load_config_and_model
from osrl.algorithms.cdt import CDT
from osrl.algorithms.ccdt import ContrastiveCDTBack

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- 1. MODEL LOADING ---
def resolve_config_path(env_name, model_type):
    if model_type == "vanilla":
        pattern = os.path.join(PROJECT_ROOT, f"models/cdt//Vanilla_CDT_Offline{env_name}-v0/**/config.yaml")
    else:
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
    env = OfflineEnvWrapper(env)
    
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
    
    return model, env, cfg

# --- 2. DATASET SAMPLING ---
def get_offline_context_windows(env, cfg, num_samples=500):
    dataset = env.get_dataset()
    obs = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    costs = dataset.get('costs', dataset.get('item_costs', np.zeros_like(rewards)))
    
    seq_len = cfg["seq_len"]
    valid_starts = len(obs) - seq_len - 1
    
    # We will sample random slices from the dataset
    indices = random.sample(range(seq_len, valid_starts), min(num_samples, valid_starts))
    
    state_seqs, action_seqs, rew_seqs, cost_seqs, time_seqs = [], [], [], [], []
    
    for idx in indices:
        s_seq = obs[idx - seq_len : idx]
        a_seq = actions[idx - seq_len : idx]
        
        # Approximate Returns and Costs to go (offline shortcut)
        rtg = np.sum(rewards[idx : idx + 100]) * cfg.get("reward_scale", 1.0)
        ctg = np.sum(costs[idx : idx + 100]) * cfg.get("cost_scale", 1.0)
        
        r_seq = np.full((seq_len,), rtg)
        c_seq = np.full((seq_len,), ctg)
        t_seq = np.arange(0, seq_len)
        
        state_seqs.append(s_seq)
        action_seqs.append(a_seq)
        rew_seqs.append(r_seq)
        cost_seqs.append(c_seq)
        time_seqs.append(t_seq)
        
    return (
        torch.tensor(np.array(state_seqs), dtype=torch.float32, device=DEVICE),
        torch.tensor(np.array(action_seqs), dtype=torch.float32, device=DEVICE),
        torch.tensor(np.array(rew_seqs), dtype=torch.float32, device=DEVICE),
        torch.tensor(np.array(cost_seqs), dtype=torch.float32, device=DEVICE),
        torch.tensor(np.array(time_seqs), dtype=torch.long, device=DEVICE)
    )

def get_model_action(model, s, a, r, c, t):
    with torch.no_grad():
        if hasattr(model, "get_action"):
            try:
                preds = model.get_action(s, a, r, c, t)
            except TypeError:
                preds = model.get_action(s, a, r, c)
            return preds
        else:
            action_preds, _, _ = model(s, a, r, c, t)
            if hasattr(action_preds, 'mean'):
                return action_preds.mean[:, -1]
            return action_preds[:, -1]

# --- 3. LIPSCHITZ EVALUATION ---
def evaluate_lipschitz(env_name, epsilons):
    results = []
    models_to_test = ["vanilla", "ccdt"]
    
    for model_type in models_to_test:
        print(f"\n🧠 Evaluating {model_type.upper()}...")
        config_path = resolve_config_path(env_name, model_type)
        model, env, cfg = load_agent(config_path, model_type)
        
        # Get a massive batch of context windows
        s_batch, a_batch, r_batch, c_batch, t_batch = get_offline_context_windows(env, cfg, num_samples=1000)
        
        # 1. Calculate Base Actions
        base_actions = get_model_action(model, s_batch, a_batch, r_batch, c_batch, t_batch)
        
        for eps_scale in epsilons:
            # 2. Generate perturbation (epsilon) for the CURRENT state (last in sequence)
            eps = torch.randn_like(s_batch[:, -1, :]) * eps_scale
            
            # Create perturbed state batch
            s_perturbed = s_batch.clone()
            s_perturbed[:, -1, :] += eps
            
            # 3. Calculate Perturbed Actions
            noisy_actions = get_model_action(model, s_perturbed, a_batch, r_batch, c_batch, t_batch)
            
            # 4. Calculate Lipschitz Constant: ||a_base - a_noisy|| / ||eps||
            action_diff_norm = torch.norm(base_actions - noisy_actions, dim=-1)
            eps_norm = torch.norm(eps, dim=-1)
            
            # Avoid division by zero
            valid_idx = eps_norm > 1e-6
            lipschitz_vals = (action_diff_norm[valid_idx] / eps_norm[valid_idx]).cpu().numpy()
            
            avg_L = np.mean(lipschitz_vals)
            std_L = np.std(lipschitz_vals)
            
            results.append({"Model": model_type.upper(), "Epsilon": eps_scale, "Avg_Lipschitz": avg_L, "Std_Lipschitz": std_L})
            print(f"  📏 Epsilon: {eps_scale:<5} | Lipschitz (L): {avg_L:.4f}")
            
    return pd.DataFrame(results)

# --- 4. PLOTTING ---
def plot_lipschitz(df, env_name):
    plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"VANILLA": "#E63946", "CCDT": "#2A9D8F"}
    
    for model in df["Model"].unique():
        subset = df[df["Model"] == model]
        ax.plot(subset["Epsilon"], subset["Avg_Lipschitz"], marker='o', linewidth=2.5, label=model, color=colors[model])
        ax.fill_between(subset["Epsilon"], subset["Avg_Lipschitz"] - (subset["Std_Lipschitz"]*0.2), 
                        subset["Avg_Lipschitz"] + (subset["Std_Lipschitz"]*0.2), alpha=0.15, color=colors[model])

    ax.set_xlabel(r"State Perturbation Magnitude ($||\epsilon||$)", fontweight="bold")
    ax.set_ylabel("Empirical Action Lipschitz Constant ($L$)", fontweight="bold")
    ax.set_title(f"Action Smoothness (Lipschitz Continuity)\n{env_name}", pad=15, fontweight="bold")
    
    # Use log scale if Vanilla explodes compared to CCDT
    ax.set_yscale('log')
    ax.legend(frameon=False)
    
    os.makedirs(os.path.join(PROJECT_ROOT, "examples/eval/eval_suite/plots"), exist_ok=True)
    out_path = os.path.join(PROJECT_ROOT, f"examples/eval/eval_suite/plots/lipschitz_{env_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Plot saved to {out_path}")

if __name__ == "__main__":
    target_envs = ["AntRun", "CarCircle", "DroneRun"]
    # Epsilon represents tiny shifts in the state. We sweep from microscopic to small.
    epsilon_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2] 
    
    for target_env in target_envs:
        print("\n" + "="*50)
        print(f"🚀 Booting Lipschitz Smoothness Evaluation for {target_env}...")
        print("="*50)
        
        try:
            df_results = evaluate_lipschitz(target_env, epsilon_range)
            print(f"\n📊 Raw Results for {target_env}:")
            print(df_results.to_string(index=False))
            plot_lipschitz(df_results, target_env)
        except Exception as e:
            print(f"❌ Skipping {target_env} due to error: {str(e)}")

    print("\n🏁 All Lipschitz evaluations complete! Check your plots directory.")