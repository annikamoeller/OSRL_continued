#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import traceback
import gymnasium as gym
import sys
import datetime
import argparse
import yaml
import random

# --- PATH SETUP ---
PROJECT_ROOT = "/home/20234949/thesis/OSRL_continued"
sys.path.insert(0, PROJECT_ROOT)

import bullet_safety_gym  # noqa
import dsrl
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.common.exp_util import load_config_and_model, seed_all

# Model Imports
from osrl.algorithms.ccdt import ContrastiveCDTFront, ContrastiveCDTBack, ContrastiveCDTTrainer
from osrl.algorithms.cdt import CDT, CDTTrainer

# --- CONSTANTS & CONFIGS ---
LOG_ROOT = {
    "ccdt": os.path.join(PROJECT_ROOT, "models"),
    "ccdt_buckets": os.path.join(PROJECT_ROOT, "models/ccdt_buckets"),
    "ccdt_cw0": os.path.join(PROJECT_ROOT, "models/ccdt_cw0"),
    "ccdt_distance": os.path.join(PROJECT_ROOT, "models/ccdt_distance"),
    "ccdt_threshold": os.path.join(PROJECT_ROOT, "models/ccdt_threshold"),
    "cdt": os.path.join(PROJECT_ROOT, "models/cdt"),
}
BASE_EVAL_DIR = os.path.join(PROJECT_ROOT, "examples/eval/eval_suite")
STATS_CSV = os.path.join(PROJECT_ROOT, "dataset_analysis/master_dataset_stats.csv")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EPSILON = 1e-8

# Evaluation Parameters
NUM_EPISODES = 20
TARGET_COST_SWEEP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
TARGET_REWARD_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.25]
NOISE_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
LIPSCHITZ_EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
ZERO_SHOT_COSTS = [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]  # Unseen interpolation targets

# Publication-Grade Visual Palette
PALETTE = {
    "CDT Baseline": "#E63946",  
    "Front - Baseline": "#F4A261",  
    "Back - Baseline": "#2A9D8F",   
    "Back-2B": "#3A8D9D",  # Added for 2 Buckets
    "Back-3B": "#457B9D",  
    "Back-5B": "#1D3557",  
    "Front-2B": "#3A8D9D",  # Added for 2 Buckets
    "Front-3B": "#457B9D",  
    "Front-5B": "#1D3557",  
    "Threshold": "#9B5DE5",  # Added for future use
    "Distance": "#F15BB5",   # Added for future use
}

# =====================================================================
# 1. CORE UTILITIES & MODEL LOADING ENGINE
# =====================================================================
def load_dataset_stats():
    if not os.path.exists(STATS_CSV):
        raise FileNotFoundError(f"❌ Missing {STATS_CSV}.")
    return pd.read_csv(STATS_CSV).set_index("Task").to_dict("index")


def construct_yaml_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", construct_yaml_tuple)


class GaussianNoiseObservationWrapper(gym.ObservationWrapper):
    """Injects Gaussian noise into state observations to simulate sensor degradation."""
    def __init__(self, env, noise_scale=0.0):
        super().__init__(env)
        self.noise_scale = noise_scale

    def observation(self, obs):
        if self.noise_scale > 0.0:
            noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=obs.shape)
            return (obs + noise).astype(np.float32)
        return obs

    def reset(self, **kwargs):
        # Bridge the gap between Gymnasium (new) and Gym (old)
        try:
            # Try passing kwargs (Modern Gymnasium API)
            result = self.env.reset(**kwargs)
        except TypeError:
            # Fallback for legacy dsrl / bullet_safety_gym API
            result = self.env.reset()
            
        # Guarantee it returns (state, info) so unpacking doesn't break
        if isinstance(result, tuple) and len(result) == 2:
            return self.observation(result[0]), result[1]
        return self.observation(result), {}


def load_model_and_env(config_path, model_type):
    """Universal loader yielding raw PyTorch models, environments, and meta-configs."""
    exp_dir = os.path.dirname(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    try:
        _, model_weights = load_config_and_model(exp_dir, best=False)
    except:
        _, model_weights = load_config_and_model(exp_dir, best=True)

    seed_all(cfg["seed"])
    base_env = gym.make(cfg["task"])
    env = wrap_env(env=base_env, reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)

    clean_task = cfg["task"].replace("Offline", "").replace("-v0", "")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    print(cfg["embedding_dim"])
    if model_type == "cdt":
        model = CDT(
            state_dim=state_dim, action_dim=action_dim, max_action=max_action,
            embedding_dim=cfg["embedding_dim"], seq_len=cfg["seq_len"],
            episode_len=cfg["episode_len"], num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"], time_emb=cfg["time_emb"],
            use_rew=cfg["use_rew"], use_cost=cfg["use_cost"],
            cost_transform=cfg["cost_transform"], target_entropy=-action_dim,
            stochastic=cfg.get("stochastic", True)
        )
        arch, variant_suffix, cw = "Vanilla", "Baseline", 0.0
    else:
        project_name = cfg.get("project", "")
        encoder_type = cfg.get("encoder_type", "front").lower()
        is_back = "back" in project_name.lower() or encoder_type == "back"
        ModelClass = ContrastiveCDTBack if is_back else ContrastiveCDTFront
        arch = "Back" if is_back else "Front"
        cw = float(cfg.get("contrastive_weight", cfg.get("args", {}).get("contrastive_weight", 0.0)))

        # --- THIS IS THE NEW LABELING FIX ---
        if "distance" in model_type.lower():
            variant_suffix = "Distance"
        elif "threshold" in model_type.lower():
            variant_suffix = "Threshold"
        else:
            buckets = cfg.get("num_buckets", cfg.get("args", {}).get("num_buckets", 1))
            variant_suffix = f"{buckets} Buckets"

        model = ModelClass(
            state_dim=state_dim, action_dim=action_dim, max_action=max_action,
            embedding_dim=cfg["embedding_dim"], contrastive_dim=cfg.get("contrastive_dim", 64),
            seq_len=cfg["seq_len"], episode_len=cfg["episode_len"], num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"], use_rew=cfg["use_rew"], use_cost=cfg["use_cost"],
            cost_transform=cfg["cost_transform"], stochastic=cfg.get("stochastic", True)
        )

    state_dict = model_weights.get("model_state", model_weights.get("model", model_weights))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Notice it returns `variant_suffix` instead of `buckets` now
    return model, env, cfg, arch, variant_suffix, cw, clean_task

def get_model_action(model, states, actions, rewards, costs, time_steps):
    """Safely extracts deterministic or stochastic mean actions across architectures."""
    with torch.no_grad():
        if hasattr(model, "get_action"):
            try:
                return model.get_action(states, actions, rewards, costs, time_steps)
            except TypeError:
                return model.get_action(states, actions, rewards, costs)
        else:
            action_preds, _, _ = model(states, actions, rewards, costs, time_steps)
            return action_preds.mean[:, -1] if hasattr(action_preds, "mean") else action_preds[:, -1]


# =====================================================================
# 2. EVALUATION EXECUTION ENGINES
# =====================================================================
def execute_rollouts(model, env, cfg, target_ret, target_cost, num_episodes=10):
    """Executes manual sliding-window context rollouts."""
    total_costs, total_rets = [], []
    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_cost, ep_ret = 0.0, 0.0

        states = torch.zeros((1, cfg["seq_len"], env.observation_space.shape[0]), device=DEVICE)
        actions = torch.zeros((1, cfg["seq_len"], env.action_space.shape[0]), device=DEVICE)
        rewards = torch.zeros((1, cfg["seq_len"]), device=DEVICE)
        costs = torch.zeros((1, cfg["seq_len"]), device=DEVICE)
        time_steps = torch.zeros((1, cfg["seq_len"]), dtype=torch.long, device=DEVICE)

        ret_to_go, cost_to_go = target_ret, target_cost

        for t in range(cfg["episode_len"]):
            states[0, -1] = torch.tensor(state, device=DEVICE)
            rewards[0, -1] = ret_to_go
            costs[0, -1] = cost_to_go
            time_steps[0, -1] = t

            action_tensor = get_model_action(model, states, actions, rewards, costs, time_steps)
            action = action_tensor.squeeze().cpu().numpy()
            next_state, reward, terminated, truncated, info = env.step(action)

            cost = info.get("cost", 0.0)
            ep_cost += cost
            ep_ret += reward

            # Advance sliding context window
            states = torch.cat([states[:, 1:], torch.zeros((1, 1, states.shape[-1]), device=DEVICE)], dim=1)
            actions[0, -1] = torch.tensor(action, device=DEVICE)
            actions = torch.cat([actions[:, 1:], torch.zeros((1, 1, actions.shape[-1]), device=DEVICE)], dim=1)
            rewards = torch.cat([rewards[:, 1:], torch.zeros((1, 1), device=DEVICE)], dim=1)
            costs = torch.cat([costs[:, 1:], torch.zeros((1, 1), device=DEVICE)], dim=1)
            time_steps = torch.cat([time_steps[:, 1:], torch.zeros((1, 1), dtype=torch.long, device=DEVICE)], dim=1)

            ret_to_go -= reward * cfg["reward_scale"]
            cost_to_go -= cost * cfg.get("cost_scale", 1.0)
            state = next_state

            if terminated or truncated:
                break

        total_costs.append(ep_cost)
        total_rets.append(ep_ret)
    return total_rets, total_costs


def run_evaluation(log_filter, eval_mode="cost_sweep", model_type="ccdt"):
    stats_lookup = load_dataset_stats()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    RUN_DIR = os.path.join(BASE_EVAL_DIR, f"{model_type}_{eval_mode}_{timestamp}")
    os.makedirs(RUN_DIR, exist_ok=True)
    output_csv = os.path.join(RUN_DIR, "raw_data.csv")

    root_dir = LOG_ROOT.get(model_type, LOG_ROOT["ccdt"])
    config_files = glob.glob(os.path.join(root_dir, log_filter, "**", "config.yaml"), recursive=True)

    if not config_files:
        print(f"❌ No models matched pattern in {root_dir}")
        return

    print(f"🔍 Executing {eval_mode.upper()} across {len(config_files)} models. Saving to: {RUN_DIR}")

    for config_path in config_files:
        try:
            model, env, cfg, arch, buckets, cw, task = load_model_and_env(config_path, model_type)
            match = next((k for k in stats_lookup.keys() if task in k), None)
            ds_max_ret = stats_lookup[match]["Return_Max"] if match else 1000.0

            variant_name = "CDT Baseline" if model_type == "cdt" else f"{arch} - {buckets} Buckets"

            # -------------------------------------------------------------
            # MODE A: Standard Cost & Pareto Sweeps
            # -------------------------------------------------------------
            if eval_mode in ["cost_sweep", "pareto_matrix"]:
                reward_targets = [ds_max_ret] if eval_mode == "cost_sweep" else [ds_max_ret * m for m in TARGET_REWARD_MULTIPLIERS]
                trainer = CDTTrainer(model, env, cfg["reward_scale"], cfg["cost_scale"], device=DEVICE) if model_type == "cdt" \
                    else ContrastiveCDTTrainer(model, env, device=DEVICE, reward_scale=cfg["reward_scale"], cost_scale=cfg["cost_scale"])

                for t_rew in reward_targets:
                    for t_cost in TARGET_COST_SWEEP:
                        print(f"  🚀 Eval | {variant_name} | Target Cost: {t_cost} | Target Rew: {t_rew:.1f}")
                        raw_ret, raw_cost, ep_len = trainer.evaluate(NUM_EPISODES, t_rew * cfg["reward_scale"], t_cost * cfg["cost_scale"])
                        raw_ret = raw_ret / cfg["reward_scale"]
                        row = {"Task": task, "Variant": variant_name, "Architecture": arch, "Buckets": buckets, "CW": cw,
                               "Target_Cost": t_cost, "Target_Reward": t_rew, "Eval_Cost": raw_cost, "Eval_Reward": raw_ret, "Ep_Len": ep_len}
                        pd.DataFrame([row]).to_csv(output_csv, mode="a", header=not os.path.exists(output_csv), index=False)

            # -------------------------------------------------------------
            # MODE B: Out-of-Distribution Sensor Noise Robustness
            # -------------------------------------------------------------
            elif eval_mode == "noise_robustness":
                target_ret = ds_max_ret * cfg["reward_scale"]
                target_cost = 10.0 * cfg.get("cost_scale", 1.0)

                for noise in NOISE_SWEEP:
                    print(f"  🌪️ Noise Level: {noise} | Variant: {variant_name}")
                    noisy_env = GaussianNoiseObservationWrapper(env, noise_scale=noise)
                    _, costs = execute_rollouts(model, noisy_env, cfg, target_ret, target_cost, NUM_EPISODES)
                    
                    row = {"Task": task, "Variant": variant_name, "Architecture": arch, "Buckets": buckets, "CW": cw,
                           "Noise_Scale": noise, "Avg_Cost": np.mean(costs), "Std_Cost": np.std(costs)}
                    pd.DataFrame([row]).to_csv(output_csv, mode="a", header=not os.path.exists(output_csv), index=False)

            # -------------------------------------------------------------
            # MODE C: Lipschitz Continuity (Action Smoothness)
            # -------------------------------------------------------------
            elif eval_mode == "lipschitz_smoothness":
                dataset = env.get_dataset()
                obs, actions = dataset["observations"], dataset["actions"]
                valid_starts = len(obs) - cfg["seq_len"] - 1
                indices = random.sample(range(cfg["seq_len"], valid_starts), min(500, valid_starts))

                s_seqs = torch.tensor(np.array([obs[i - cfg["seq_len"]: i] for i in indices]), dtype=torch.float32, device=DEVICE)
                a_seqs = torch.tensor(np.array([actions[i - cfg["seq_len"]: i] for i in indices]), dtype=torch.float32, device=DEVICE)
                r_seqs = torch.full((len(indices), cfg["seq_len"]), ds_max_ret * cfg["reward_scale"], device=DEVICE)
                c_seqs = torch.full((len(indices), cfg["seq_len"]), 10.0 * cfg.get("cost_scale", 1.0), device=DEVICE)
                t_seqs = torch.arange(0, cfg["seq_len"], device=DEVICE).repeat(len(indices), 1)

                base_actions = get_model_action(model, s_seqs, a_seqs, r_seqs, c_seqs, t_seqs)

                for eps_scale in LIPSCHITZ_EPSILONS:
                    print(f"  📏 Lipschitz | Epsilon: {eps_scale} | Variant: {variant_name}")
                    eps = torch.randn_like(s_seqs[:, -1, :]) * eps_scale
                    s_perturbed = s_seqs.clone()
                    s_perturbed[:, -1, :] += eps

                    noisy_actions = get_model_action(model, s_perturbed, a_seqs, r_seqs, c_seqs, t_seqs)
                    diff_norm = torch.norm(base_actions - noisy_actions, dim=-1)
                    eps_norm = torch.norm(eps, dim=-1)

                    valid = eps_norm > 1e-6
                    L_vals = (diff_norm[valid] / eps_norm[valid]).cpu().numpy()

                    row = {"Task": task, "Variant": variant_name, "Architecture": arch, "Buckets": buckets, "CW": cw,
                           "Epsilon": eps_scale, "Avg_Lipschitz": np.mean(L_vals), "Std_Lipschitz": np.std(L_vals)}
                    pd.DataFrame([row]).to_csv(output_csv, mode="a", header=not os.path.exists(output_csv), index=False)

            # -------------------------------------------------------------
            # MODE D: Zero-Shot Generalization (Cost Interpolation)
            # -------------------------------------------------------------
            elif eval_mode == "zeroshot_interpolation":
                target_ret = ds_max_ret * cfg["reward_scale"]

                for target_c in ZERO_SHOT_COSTS:
                    print(f"  🎯 Zero-Shot | Target Cost: {target_c} | Variant: {variant_name}")
                    _, costs = execute_rollouts(model, env, cfg, target_ret, target_c * cfg.get("cost_scale", 1.0), NUM_EPISODES)
                    mae = np.mean(np.abs(np.array(costs) - target_c))

                    row = {"Task": task, "Variant": variant_name, "Architecture": arch, "Buckets": buckets, "CW": cw,
                           "Target_Cost": target_c, "MAE": mae}
                    pd.DataFrame([row]).to_csv(output_csv, mode="a", header=not os.path.exists(output_csv), index=False)

        except Exception as e:
            print(f"❌ Execution failed on {config_path}:")
            traceback.print_exc()

    print(f"\n✅ {eval_mode.upper()} evaluation finished successfully! Results saved to {output_csv}")


# =====================================================================
# 3. PUBLICATION-GRADE VISUALIZATION ENGINE
# =====================================================================
def set_professional_style():
    """Configures professional typography, layout, and minimal chart junk."""
    sns.set_theme(style="whitegrid", font="sans-serif", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.2, "grid.linestyle": "--", "grid.alpha": 0.7
    })


def load_and_merge_data(run_dir, vanilla_csv=None):
    df = pd.read_csv(os.path.join(run_dir, "raw_data.csv"))
    if vanilla_csv and os.path.exists(vanilla_csv):
        v_df = pd.read_csv(vanilla_csv)
        v_df["Variant"] = "CDT Baseline"
        df = pd.concat([df, v_df], ignore_index=True)
        # --- DATA NORMALIZATION (Fixes inconsistent naming) ---
        df = df.rename(columns={
        "Raw_Eval_Cost": "Eval_Cost", 
        "Raw_Eval_Reward": "Eval_Reward",
        "Contrastive_Weight": "CW"
        })
        
        cleanup_map = {
            "Back - 2 Buckets": "Back-2B",
            "Back - 3 Buckets": "Back-3B",
            "Back - 5 Buckets": "Back-5B",
            "Front - 2 Buckets": "Front-2B",
            "Front - 3 Buckets": "Front-3B",
            "Front - 5 Buckets": "Front-5B",
            "Back - Distance Buckets": "Distance",
            "Back - Threshold Buckets": "Threshold",
            "Vanilla Baseline": "CDT Baseline",
            "Vanilla": "CDT Baseline"
        }
        
        # Check if the 'Variant' column exists, then apply the cleanup
        if "Variant" in df.columns:
            df["Variant"] = df["Variant"].replace(cleanup_map)
            
        print("data loaded and merged")
    return df


def plot_cost_adherence(run_dir, vanilla_csv=None):
    set_professional_style()
    df = load_and_merge_data(run_dir, vanilla_csv)
    tasks = sorted(df["Task"].unique())
    fig, axes = plt.subplots(2, len(tasks), figsize=(5 * len(tasks), 8), sharex=True)
    if len(tasks) == 1: axes = axes.reshape(2, 1)

    for i, task in enumerate(tasks):
        sub = df[df["Task"] == task]
        sns.lineplot(ax=axes[0, i], data=sub, x="Target_Cost", y="Eval_Reward", hue="Variant", palette=PALETTE, marker="o", linewidth=2.2, legend=(i == len(tasks)-1))
        axes[0, i].set_title(task, fontweight="bold", pad=10)
        axes[0, i].set_ylabel("Evaluated Return" if i == 0 else "")

        sns.lineplot(ax=axes[1, i], data=sub, x="Target_Cost", y="Eval_Cost", hue="Variant", palette=PALETTE, marker="s", linewidth=2.2, legend=False)
        sweep = sorted(sub["Target_Cost"].unique())
        axes[1, i].plot(sweep, sweep, "k:", alpha=0.6, label="Ideal Adherence")
        axes[1, i].axhline(10.0, color="r", linestyle="--", alpha=0.5)
        axes[1, i].set_ylabel("Evaluated Cost" if i == 0 else "")
        axes[1, i].set_xlabel("Target Cost Prompt")

    if len(tasks) > 0: axes[0, -1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plot_cost_adherence.png"), dpi=300, bbox_inches="tight")


def plot_pareto_frontier(run_dir, vanilla_csv=None):
    set_professional_style()
    df = load_and_merge_data(run_dir, vanilla_csv)
    tasks = sorted(df["Task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5))
    if len(tasks) == 1: axes = [axes]

    for i, task in enumerate(tasks):
        sub = df[df["Task"] == task]
        sns.scatterplot(ax=axes[i], data=sub, x="Eval_Cost", y="Eval_Reward", hue="Variant", palette=PALETTE, alpha=0.3, legend=False)
        
        for var in sub["Variant"].unique():
            v_df = sub[sub["Variant"] == var].sort_values("Eval_Cost")
            v_df["Pareto_Max"] = v_df["Eval_Reward"].cummax()
            sns.lineplot(ax=axes[i], data=v_df, x="Eval_Cost", y="Pareto_Max", color=PALETTE.get(var, "#000"), linewidth=2.5, label=var)

        axes[i].set_title(f"{task} (Pareto Frontier)", fontweight="bold")
        axes[i].axvline(10.0, color="r", linestyle="--", alpha=0.5)
        axes[i].set_xlabel("Evaluated Cost (Lower is Safer)")
        axes[i].set_ylabel("Evaluated Return (Higher is Better)" if i == 0 else "")
        if i == len(tasks)-1: axes[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else: axes[i].get_legend().remove()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plot_pareto_frontier.png"), dpi=300, bbox_inches="tight")


def plot_noise_robustness(run_dir, vanilla_csv=None):
    set_professional_style()
    df = load_and_merge_data(run_dir, vanilla_csv)
    tasks = sorted(df["Task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5))
    
    # Handle single-task evaluation gracefully
    if len(tasks) == 1: 
        axes = [axes]

    for i, task in enumerate(tasks):
        sub = df[df["Task"] == task]
        for var in sub["Variant"].unique():
            
            # --- THE FIX: Aggregate any duplicate rows and strictly sort by X-axis ---
            v_df = sub[sub["Variant"] == var]
            # Group by Noise_Scale and average everything else, then sort left-to-right
            v_df = v_df.groupby("Noise_Scale", as_index=False).mean(numeric_only=True)
            v_df = v_df.sort_values("Noise_Scale")
            
            axes[i].plot(v_df["Noise_Scale"], v_df["Avg_Cost"], marker="o", linewidth=2.5, label=var, color=PALETTE.get(var, "#000"))
            axes[i].fill_between(v_df["Noise_Scale"], v_df["Avg_Cost"] - v_df["Std_Cost"], v_df["Avg_Cost"] + v_df["Std_Cost"], alpha=0.15, color=PALETTE.get(var, "#000"))

        # Formatting
        axes[i].axhline(10.0, color="r", linestyle="--", alpha=0.8, label="Safety Limit" if i==0 else "")
        axes[i].set_title(f"Noise Robustness: {task}", fontweight="bold")
        axes[i].set_xlabel("State Noise (Gaussian Std Dev)")
        axes[i].set_ylabel("Average Trajectory Cost" if i == 0 else "")
        if i == len(tasks)-1: 
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plot_noise_robustness.png"), dpi=300, bbox_inches="tight")


def plot_lipschitz_continuity(run_dir, vanilla_csv=None):
    set_professional_style()
    df = load_and_merge_data(run_dir, vanilla_csv)
    tasks = sorted(df["Task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5))
    if len(tasks) == 1: axes = [axes]

    for i, task in enumerate(tasks):
        sub = df[df["Task"] == task]
        for var in sub["Variant"].unique():
            v_df = sub[sub["Variant"] == var]
            
            # --- THE FIX: Aggregate the seeds so Matplotlib draws one clean line ---
            v_df = v_df.groupby("Epsilon", as_index=False).mean(numeric_only=True)
            v_df = v_df.sort_values("Epsilon")

            axes[i].plot(v_df["Epsilon"], v_df["Avg_Lipschitz"], marker="o", linewidth=2.5, label=var, color=PALETTE.get(var, "#000"))
            axes[i].fill_between(v_df["Epsilon"], v_df["Avg_Lipschitz"] - v_df["Std_Lipschitz"]*0.2, v_df["Avg_Lipschitz"] + v_df["Std_Lipschitz"]*0.2, alpha=0.15, color=PALETTE.get(var, "#000"))

        axes[i].set_title(f"Action Smoothness: {task}", fontweight="bold")
        axes[i].set_xlabel("Perturbation Magnitude (Epsilon)")
        axes[i].set_ylabel("Empirical Lipschitz Constant (L)" if i == 0 else "")
        axes[i].set_yscale("log")
        if i == len(tasks)-1: axes[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plot_lipschitz_continuity.png"), dpi=300, bbox_inches="tight")


def plot_zeroshot_generalization(run_dir, vanilla_csv=None):
    set_professional_style()
    df = load_and_merge_data(run_dir, vanilla_csv)
    tasks = sorted(df["Task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5))
    if len(tasks) == 1: axes = [axes]

    for i, task in enumerate(tasks):
        sub = df[df["Task"] == task]
        sns.lineplot(ax=axes[i], data=sub, x="Target_Cost", y="MAE", hue="Variant", palette=PALETTE, marker="s", linewidth=2.5)
        axes[i].set_title(f"Zero-Shot Generalization: {task}", fontweight="bold")
        axes[i].set_xlabel("Unseen Target Cost Prompt")
        axes[i].set_ylabel("Mean Absolute Error (Cost)" if i == 0 else "")
        if i == len(tasks)-1: axes[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else: axes[i].get_legend().remove()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plot_zeroshot_generalization.png"), dpi=300, bbox_inches="tight")


# =====================================================================
# CLI ROUTER
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Publication-Grade Safe RL Suite")
    parser.add_argument("mode", choices=[
        "eval_cost", "eval_pareto", "eval_noise", "eval_lipschitz", "eval_zeroshot",
        "plot_cost", "plot_pareto", "plot_noise", "plot_lipschitz", "plot_zeroshot"
    ], help="Select operational engine or visualizer.")

    parser.add_argument("--model_type", type=str, default="ccdt", choices=list(LOG_ROOT.keys()), help="Target directory mapping.")
    parser.add_argument("--log_filter", type=str, default="*_cw*", help="Glob wildcard pattern.")
    parser.add_argument("--run_dir", type=str, help="Required path to timestamped evaluation directory for plotting.")
    parser.add_argument("--vanilla_csv", type=str, default=None, help="Optional baseline CSV to inject into charts.")
    args = parser.parse_args()

    # Routing matrix
    if args.mode.startswith("eval_"):
        mode_map = {"eval_cost": "cost_sweep", "eval_pareto": "pareto_matrix", "eval_noise": "noise_robustness",
                    "eval_lipschitz": "lipschitz_smoothness", "eval_zeroshot": "zeroshot_interpolation"}
        run_evaluation(args.log_filter, eval_mode=mode_map[args.mode], model_type=args.model_type)
    else:
        if not args.run_dir: raise ValueError("❌ --run_dir argument required for plotting modes.")
        plot_map = {"plot_cost": plot_cost_adherence, "plot_pareto": plot_pareto_frontier, "plot_noise": plot_noise_robustness,
                    "plot_lipschitz": plot_lipschitz_continuity, "plot_zeroshot": plot_zeroshot_generalization}
        plot_map[args.mode](args.run_dir, args.vanilla_csv)