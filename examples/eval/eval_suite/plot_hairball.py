#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import seaborn as sns
from sklearn.manifold import TSNE
import gymnasium as gym
import yaml
import glob
import random
import argparse
import umap

# --- PATH SETUP ---
PROJECT_ROOT = "/home/20234949/thesis/OSRL_continued"
sys.path.insert(0, PROJECT_ROOT)

import bullet_safety_gym  # noqa
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.common.exp_util import load_config_and_model
from osrl.algorithms.cdt import CDT
from osrl.algorithms.ccdt import ContrastiveCDTBack

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- CORE FUNCTIONS ---
def resolve_config_path(env_name, model_type):
    if model_type == "vanilla":
        pattern = os.path.join(PROJECT_ROOT, f"models/cdt//Vanilla_CDT_Offline{env_name}-v0/**/config.yaml")
    elif model_type == "ccdt":
        pattern = os.path.join(PROJECT_ROOT, f"models/ccdt_arch_a/Back_Offline{env_name}-v0_3Buckets_cw03/**/config.yaml")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"❌ Could not find config for {model_type} {env_name} matching: {pattern}")
    return matches[0]

hidden_states_buffer = []

def hidden_state_hook(module, inp, out):
    hidden = inp[0].detach().cpu()
    if hidden.dim() == 3:
        hidden = hidden[:, -1, :] 
    hidden_states_buffer.append(hidden.numpy())

def load_evaluation_model(config_path, model_type):
    exp_dir = os.path.dirname(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.unsafe_load(f)

    try:
        _, model_weights = load_config_and_model(exp_dir, best=False)
    except:
        _, model_weights = load_config_and_model(exp_dir, best=True)

    base_env = gym.make(cfg["task"])
    env = wrap_env(env=base_env, reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)

    if model_type == "vanilla":
        model = CDT(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            max_action=env.action_space.high[0],
            embedding_dim=cfg["embedding_dim"],
            seq_len=cfg["seq_len"],
            episode_len=cfg["episode_len"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            time_emb=cfg["time_emb"],
            use_rew=cfg["use_rew"],
            use_cost=cfg["use_cost"],
            cost_transform=cfg["cost_transform"],
            target_entropy=-env.action_space.shape[0],
            stochastic=cfg.get("stochastic", True), 
        )
    else:
        model = ContrastiveCDTBack(
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
            stochastic=cfg.get("stochastic", True), 
        )

    state_dict = model_weights.get("model_state", model_weights.get("model", model_weights))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    for name, module in model.named_modules():
        if "predict_action" in name or "action_head" in name:
            module.register_forward_hook(hidden_state_hook)
            break

    return model, env, cfg

def get_offline_trajectories(env, num_trajs=500):
    dataset = env.get_dataset()
    obs = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    costs = dataset.get('costs', dataset.get('item_costs', np.zeros_like(rewards)))
    terminals = dataset['terminals']
    timeouts = dataset.get('timeouts', np.zeros_like(terminals))
    
    trajectories = []
    current_traj = {'obs': [], 'acts': [], 'rews': [], 'costs': []}
    
    for i in range(len(obs)):
        current_traj['obs'].append(obs[i])
        current_traj['acts'].append(actions[i])
        current_traj['rews'].append(rewards[i])
        current_traj['costs'].append(costs[i])
        
        if terminals[i] or timeouts[i]:
            if len(current_traj['obs']) > 20: 
                trajectories.append({k: np.array(v) for k, v in current_traj.items()})
            current_traj = {'obs': [], 'acts': [], 'rews': [], 'costs': []}
            
    return random.sample(trajectories, min(num_trajs, len(trajectories)))

def extract_offline_embeddings(model, trajectories, cfg):
    global hidden_states_buffer
    hidden_states_buffer = []
    trajectory_costs = []
    seq_len = cfg["seq_len"]

    for traj in trajectories:
        t_len = len(traj['obs'])
        if t_len <= seq_len:
            continue
            
        step = random.randint(seq_len, t_len - 1)
        trajectory_costs.append(np.sum(traj['costs']))

        s_seq = torch.tensor(traj['obs'][step - seq_len : step], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a_seq = torch.tensor(traj['acts'][step - seq_len : step], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        rtg = np.sum(traj['rews'][step - seq_len :]) * cfg.get("reward_scale", 1.0)
        ctg = np.sum(traj['costs'][step - seq_len :]) * cfg.get("cost_scale", 1.0)
        
        r_seq = torch.full((1, seq_len), rtg, dtype=torch.float32, device=DEVICE)
        c_seq = torch.full((1, seq_len), ctg, dtype=torch.float32, device=DEVICE)
        t_seq = torch.arange(step - seq_len, step, dtype=torch.long, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            if hasattr(model, "get_action"):
                try:
                    _ = model.get_action(s_seq, a_seq, r_seq, c_seq, t_seq)
                except TypeError:
                    _ = model.get_action(s_seq, a_seq, r_seq, c_seq)
            else:
                _ = model(s_seq, a_seq, r_seq, c_seq, t_seq)

    embeddings = np.concatenate(hidden_states_buffer, axis=0)
    return embeddings, trajectory_costs

def plot_gradient_manifold(df, env_name, model_type, out_directory):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Roboto", "sans-serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    })

    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    scatter = ax.scatter(
        df["x"], df["y"], 
        c=df["Trajectory_Cost"], 
        cmap="RdYlGn_r", 
        s=85, alpha=0.9, edgecolors='w', linewidth=0.6
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Total Trajectory Cost", rotation=270, labelpad=20, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    title_prefix = "CDT" if model_type == "vanilla" else "Contrastive CDT"
    ax.set_title(f"{title_prefix} Latent Representation\n{env_name}", pad=15)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(out_directory, exist_ok=True)
    out_path = os.path.join(out_directory, f"{model_type}_gradient_{env_name.lower()}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Plot saved: {out_path}")

# --- CLI ENGINE ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Latent Gradients for OSRL Models")
    parser.add_argument("--models", nargs="+", choices=["vanilla", "ccdt", "both"], default=["both"], help="Models to evaluate")
    parser.add_argument("--envs", nargs="+", default=["AntRun", "CarCircle", "DroneRun"], help="Environments to evaluate")
    parser.add_argument("--trajectories", type=int, default=200, help="Number of offline trajectories to sample")
    args = parser.parse_args()

    models_to_run = ["vanilla", "ccdt"] if "both" in args.models else args.models

    print(f"🚀 Initializing Evaluation Pipeline on {DEVICE.upper()}")
    print(f"🎯 Environments: {args.envs}")
    print(f"🧠 Models: {models_to_run}")

    for env_name in args.envs:
        for m_type in models_to_run:
            print(f"\n🌍 === PROCESSING: {env_name} | MODEL: {m_type.upper()} ===")
            try:
                config_path = resolve_config_path(env_name, m_type)
                print(f"📂 Found Config: {config_path}")
                
                model, env, cfg = load_evaluation_model(config_path, m_type)
                trajectories = get_offline_trajectories(env, num_trajs=args.trajectories)
                
                print(f"🧠 Extracting {len(trajectories)} representations...")
                embeddings, costs = extract_offline_embeddings(model, trajectories, cfg)
                
                # print("🌌 Projecting to 2D via t-SNE...")
                # tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
                # embeddings_2d = tsne.fit_transform(embeddings)
                print("🌌 Projecting to 2D via UMAP...")
                reducer = umap.UMAP(
                    n_neighbors=30,
                    min_dist=0.1,
                    metric='euclidean',
                    random_state=42
                )
                embeddings_2d = reducer.fit_transform(embeddings)

                df_plot = pd.DataFrame({"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "Trajectory_Cost": costs})

                out_directory = os.path.join(PROJECT_ROOT, "examples/eval/eval_suite/plots/hairball_plots")
                plot_gradient_manifold(df_plot, env_name, m_type, out_directory)
                
            except Exception as e:
                print(f"❌ Error processing {env_name} with {m_type}: {str(e)}")
                continue