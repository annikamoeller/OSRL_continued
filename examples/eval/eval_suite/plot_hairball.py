#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import gymnasium as gym
import yaml

# --- PATH SETUP ---
PROJECT_ROOT = "/home/20234949/thesis/OSRL_continued"
sys.path.insert(0, PROJECT_ROOT)

import bullet_safety_gym  # noqa
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.algorithms.ccdt import ContrastiveCDTBack
from osrl.algorithms.cdt import CDT

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

EXPERIMENTS = {
    "AntRun": {
        "Baseline": "models/cdt/Vanilla_CDT_OfflineAntRun-v0/CDT_eval_every5000_seed8-205d/CDT_eval_every5000_seed8-205d/config.yaml",
        "CCDT_2B":  "models/ccdt_buckets/Back_OfflineAntRun-v0_2Buckets_cw03/AntRun_2B_0Pre_20260607_062913_205876/AntRun_2B_0Pre_20260607_062913_205876/config.yaml",
        "CCDT_3B":  "models/ccdt_buckets/Back_OfflineAntRun-v0_3Buckets_cw03/AntRun_3B_0Pre_20260607_123813_292781/AntRun_3B_0Pre_20260607_123813_292781/config.yaml",
        "CCDT_5B":  "models/ccdt_buckets/Back_OfflineAntRun-v0_5Buckets_cw03/AntRun_5B_0Pre_20260607_180349_176723/AntRun_5B_0Pre_20260607_180349_176723/config.yaml",
    },
    "CarCircle": {
        "Baseline": "models/cdt/Vanilla_CDT_OfflineCarCircle-v0/CDT_eval_every5000_seed8-4818/CDT_eval_every5000_seed8-4818/config.yaml",
        "CCDT_2B":  "models/ccdt_buckets/Back_OfflineCarCircle-v0_2Buckets_cw03/CarCircle_2B_0Pre_20260608_043629_697766/CarCircle_2B_0Pre_20260608_043629_697766/config.yaml",
        "CCDT_3B":  "models/ccdt_buckets/Back_OfflineCarCircle-v0_3Buckets_cw03/CarCircle_3B_0Pre_20260608_080821_878539/CarCircle_3B_0Pre_20260608_080821_878539/config.yaml",
        "CCDT_5B":  "models/ccdt_buckets/Back_OfflineCarCircle-v0_5Buckets_cw03/CarCircle_5B_0Pre_20260611_152557_434367/CarCircle_5B_0Pre_20260611_152557_434367/config.yaml",
    },
    "DroneRun": {
        "Baseline": "models/cdt/Vanilla_CDT_OfflineDroneRun-v0/CDT_eval_every5000_seed8-ca68/CDT_eval_every5000_seed8-ca68/config.yaml",
        "CCDT_2B":  "models/ccdt_buckets/Back_OfflineDroneRun-v0_2Buckets_cw03/DroneRun_2B_0Pre_20260608_184340_151194/DroneRun_2B_0Pre_20260608_184340_151194/config.yaml",
        "CCDT_3B":  "models/ccdt_buckets/Back_OfflineDroneRun-v0_3Buckets_cw03/DroneRun_3B_0Pre_20260608_223145_325235/DroneRun_3B_0Pre_20260608_223145_325235/config.yaml",
        "CCDT_5B":  "models/ccdt_buckets/Back_OfflineDroneRun-v0_5Buckets_cw03/DroneRun_5B_0Pre_20260611_183416_814721/DroneRun_5B_0Pre_20260611_183416_814721/config.yaml",
    }
}

# EXPERIMENTS = {
#     "AntRun": {
#         "Baseline": "models/cdt/Vanilla_CDT_OfflineAntRun-v0/CDT_eval_every5000_seed8-205d/CDT_eval_every5000_seed8-205d/config.yaml",
#         "Distance":  "models/ccdt_distance/AntRun_Dist_a0.02_0Pre_20260622_181550/AntRun_Dist_a0.02_0Pre_20260622_181550/config.yaml",
#     },
#     "CarCircle": {
#         "Baseline": "models/cdt/Vanilla_CDT_OfflineCarCircle-v0/CDT_eval_every5000_seed8-4818/CDT_eval_every5000_seed8-4818/config.yaml",
#         "Distance":  "models/ccdt_distance/CarCircle_Dist_a0.02_0Pre_20260622_164159/CarCircle_Dist_a0.02_0Pre_20260622_164159/config.yaml",
#     },
#     "DroneRun": {
#         "Baseline": "models/cdt/Vanilla_CDT_OfflineDroneRun-v0/CDT_eval_every5000_seed8-ca68/CDT_eval_every5000_seed8-ca68/config.yaml",
#         "Distance":  "models/ccdt_distance/DroneRun_Dist_a0.02_0Pre_20260622_194357/DroneRun_Dist_a0.02_0Pre_20260622_194357/config.yaml",
#     }
# }


ROW_ORDER = ["Baseline", "CCDT_2B", "CCDT_3B", "CCDT_5B"]
ROW_LABELS = ["Vanilla CDT\n(Baseline)", "CCDT (cw=0.3)\n2 Buckets", "CCDT (cw=0.3)\n3 Buckets", "CCDT (cw=0.3)\n5 Buckets"]
# ROW_ORDER = ["Baseline", "Distance"]
# ROW_LABELS = ["Baseline", "Distance"]
ENV_ORDER = list(EXPERIMENTS.keys())

# 1. WIRETAP
hidden_states_buffer = []

def hidden_state_hook(module, inp, out):
    hidden = inp[0].detach().cpu()
    if hidden.dim() == 3:
        hidden = hidden[:, -1, :]
    hidden_states_buffer.append(hidden.numpy())

# 2. MODEL LOADING
def load_and_hook_model(config_path, is_vanilla=True):
    config_path = os.path.abspath(os.path.join(PROJECT_ROOT, config_path))
    exp_dir = os.path.dirname(config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
        
    with open(config_path, "r") as f:
        cfg = yaml.full_load(f)

    for p in ["checkpoint/model.pt", "checkpoint/model_best.pt", "model.pt", "model_best.pt"]:
        full_p = os.path.join(exp_dir, p)
        if os.path.exists(full_p):
            model_weights = torch.load(full_p, map_location=DEVICE)
            break

    base_env = gym.make(cfg["task"])
    env = wrap_env(env=base_env, reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)

    model_class = CDT if is_vanilla else ContrastiveCDTBack
    kwargs = {
        "state_dim": env.observation_space.shape[0], "action_dim": env.action_space.shape[0],
        "max_action": env.action_space.high[0], "embedding_dim": cfg["embedding_dim"],
        "seq_len": cfg["seq_len"], "episode_len": cfg["episode_len"], "num_layers": cfg["num_layers"],
        "num_heads": cfg["num_heads"], "use_rew": cfg["use_rew"], "use_cost": cfg["use_cost"],
        "cost_transform": cfg["cost_transform"], "stochastic": cfg.get("stochastic", True)
    }
    if is_vanilla:
        kwargs.update({"time_emb": cfg["time_emb"], "target_entropy": -env.action_space.shape[0]})
    else:
        kwargs.update({"contrastive_dim": cfg.get("contrastive_dim", 64)})

    model = model_class(**kwargs)
    state_dict = model_weights.get("model_state", model_weights.get("model", model_weights))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    hook_attached = False
    for name, module in model.named_modules():
        if "predict_action" in name or "action_head" in name:
            module.register_forward_hook(hidden_state_hook)
            hook_attached = True
            break
    if not hook_attached:
        model.register_forward_hook(hidden_state_hook)

    return model, env, cfg

def extract_trajectories(env):
    dataset = env.get_dataset()
    obs, acts, rews = dataset['observations'], dataset['actions'], dataset['rewards']
    costs = dataset.get('costs', np.zeros_like(rews))
    
    # 👈 FIX: Split into two separate lines
    terms = dataset['terminals']
    truncs = dataset.get('timeouts', np.zeros_like(terms))

    trajectories = []
    curr = {'obs': [], 'acts': [], 'rews': [], 'costs': []}
    
    for i in range(len(obs)):
        curr['obs'].append(obs[i])
        curr['acts'].append(acts[i])
        curr['rews'].append(rews[i])
        curr['costs'].append(costs[i])

        if terms[i] or truncs[i] or (i == len(obs) - 1):
            r_arr, c_arr = np.array(curr['rews']), np.array(curr['costs'])
            rtg, ctg = np.zeros_like(r_arr), np.zeros_like(c_arr)
            rtg[-1], ctg[-1] = r_arr[-1], c_arr[-1]
            for t in reversed(range(len(r_arr) - 1)):
                rtg[t], ctg[t] = r_arr[t] + rtg[t+1], c_arr[t] + ctg[t+1]
                
            trajectories.append({
                'obs': np.array(curr['obs']), 'acts': np.array(curr['acts']),
                'rtg': rtg, 'ctg': ctg, 'ep_cost': np.sum(c_arr), 'length': len(r_arr)
            })
            curr = {'obs': [], 'acts': [], 'rews': [], 'costs': []}
            
    return trajectories

def collect_continuous_embeddings(model, env, cfg, num_samples=1000):
    global hidden_states_buffer
    hidden_states_buffer = []  
    costs = []
    
    trajectories = extract_trajectories(env)
    
    # Just grab a massive random handful of the dataset
    chosen_idx = np.random.choice(len(trajectories), size=min(num_samples, len(trajectories)), replace=False)
    seq_len = cfg["seq_len"]
    
    for idx in chosen_idx:
        traj = trajectories[idx]
        traj_len = traj['length']
        
        t_end = np.random.randint(min(10, traj_len), traj_len) if traj_len > 10 else traj_len - 1
        t_start = max(0, t_end - seq_len + 1)
        
        s_slice = traj['obs'][t_start : t_end + 1]
        a_slice = traj['acts'][t_start : t_end + 1]
        rtg_slice = traj['rtg'][t_start : t_end + 1] * cfg["reward_scale"]
        ctg_slice = traj['ctg'][t_start : t_end + 1] * cfg["cost_scale"]
        t_slice = np.arange(t_start, t_end + 1)
        
        actual_len = s_slice.shape[0]
        
        s_seq = torch.zeros((1, seq_len, s_slice.shape[1]), dtype=torch.float32, device=DEVICE)
        a_seq = torch.zeros((1, seq_len, a_slice.shape[1]), dtype=torch.float32, device=DEVICE)
        r_seq = torch.zeros((1, seq_len), dtype=torch.float32, device=DEVICE)
        c_seq = torch.zeros((1, seq_len), dtype=torch.float32, device=DEVICE)
        time_seq = torch.zeros((1, seq_len), dtype=torch.long, device=DEVICE)
        
        s_seq[0, -actual_len:] = torch.tensor(s_slice, device=DEVICE)
        a_seq[0, -actual_len:] = torch.tensor(a_slice, device=DEVICE)
        r_seq[0, -actual_len:] = torch.tensor(rtg_slice, device=DEVICE)
        c_seq[0, -actual_len:] = torch.tensor(ctg_slice, device=DEVICE)
        time_seq[0, -actual_len:] = torch.tensor(t_slice, device=DEVICE)

        with torch.no_grad():
            if hasattr(model, "get_action"): _ = model.get_action(s_seq, a_seq, r_seq, c_seq, time_seq)
            else: _ = model(s_seq, a_seq, r_seq, c_seq, time_seq)
            
        costs.append(traj['ep_cost'])
                
    embeddings = np.concatenate(hidden_states_buffer, axis=0)
    return embeddings, np.array(costs)

# 4. EXECUTION
if __name__ == "__main__":
    print("🚀 Extracting True Offline Representations (Continuous Gradient)...")
    
    num_rows, num_cols = len(ROW_ORDER), len(ENV_ORDER)
    plot_data = {env: {} for env in ENV_ORDER}

    for col_idx, env_name in enumerate(ENV_ORDER):
        for row_idx, arch_key in enumerate(ROW_ORDER):
            config_path = EXPERIMENTS[env_name].get(arch_key)
            if not config_path or not os.path.exists(os.path.join(PROJECT_ROOT, config_path)):
                continue
                
            print(f"\n🧠 Processing {env_name} | {arch_key}...")
            is_vanilla = (arch_key == "Baseline")
            
            model, env, cfg = load_and_hook_model(config_path, is_vanilla=is_vanilla)
            embeddings, costs = collect_continuous_embeddings(model, env, cfg, num_samples=1000)
            
            print(f"🌌 Running t-SNE for {env_name} - {arch_key}...")
            tsne = TSNE(n_components=2, perplexity=40, random_state=42)
            emb_tsne = tsne.fit_transform(embeddings)
            
            plot_data[env_name][arch_key] = pd.DataFrame({"x": emb_tsne[:, 0], "y": emb_tsne[:, 1], "Cost": costs})

    print("\n🎨 Rendering continuous gradient grid...")
    sns.set_theme(style="white", font_scale=1.1)
    plt.rcParams.update({"font.family": "serif"})

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), sharex=False, sharey=False)

    for col_idx, env_name in enumerate(ENV_ORDER):
        for row_idx, arch_key in enumerate(ROW_ORDER):
            ax = axes[row_idx, col_idx]
            
            if arch_key in plot_data[env_name]:
                df = plot_data[env_name][arch_key]
                # Matplotlib automatically scales 'c' from min to max within each subplot
                ax.scatter(df["x"], df["y"], c=df["Cost"], cmap="RdYlGn_r", s=30, alpha=0.8, edgecolors='none')
            else:
                ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', color='gray')
                
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values(): 
                spine.set_color("#bdc3c7")
                
            if row_idx == 0: 
                ax.set_title(env_name, fontweight="bold", pad=15, fontsize=16)
            if col_idx == 0: 
                ax.set_ylabel(ROW_LABELS[row_idx], fontweight="bold", labelpad=15, fontsize=14)
            if row_idx == num_rows - 1: 
                ax.set_xlabel("t-SNE Dim 1", labelpad=10)

    # --- BULLETPROOF COLORBAR ---
    # Create a standalone color mapping decoupled from the subplots
    plt.tight_layout()
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([]) # Required for matplotlib

    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    
    # Set relative labels instead of raw numbers to account for different environments
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['Min Cost\n(Safe)', 'Medium Risk', 'Max Cost\n(Unsafe)'])
    cbar.set_label("Relative Episodic Cost", fontweight='bold', fontsize=14, labelpad=10)

    plt.subplots_adjust(bottom=0.15) # Give slightly more room for the new text
    
    out_path = os.path.join(PROJECT_ROOT, "examples/eval/eval_suite/tsne_continuous_grid.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Continuous Grid Plot generated successfully: {out_path}")