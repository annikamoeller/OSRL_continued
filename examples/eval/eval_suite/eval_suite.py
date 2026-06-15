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
    "ccdt": os.path.join(PROJECT_ROOT, "thesis_final_models"),
    "vanilla": os.path.join(PROJECT_ROOT, "output_cdt"),
}
BASE_EVAL_DIR = os.path.join(PROJECT_ROOT, "examples/eval/eval_suite")
STATS_CSV = os.path.join(PROJECT_ROOT, "dataset_analysis/master_dataset_stats.csv")
EPSILON = 1e-8

TARGET_COST_SWEEP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
TARGET_REWARD_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.25]  # For Pareto matrix sweeps
NUM_EPISODES = 20
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# =====================================================================
# 1. DATA PROCESSING UTILS
# =====================================================================
def load_dataset_stats():
    if not os.path.exists(STATS_CSV):
        raise FileNotFoundError(f"❌ Missing {STATS_CSV}.")
    stats_df = pd.read_csv(STATS_CSV)
    return stats_df.set_index("Task").to_dict("index")


def load_and_normalize_data(raw_data_csv, stats_lookup):
    if not os.path.exists(raw_data_csv):
        raise FileNotFoundError(f"Missing {raw_data_csv}.")
    raw_df = pd.read_csv(raw_data_csv)

    processed_records = []
    for _, row in raw_df.iterrows():
        task_name = str(row["Task"]).replace("Offline", "").replace("-v0", "")
        match = next((k for k in stats_lookup.keys() if task_name in k), None)

        if match:
            r_max = stats_lookup[match]["Return_Max"]
            r_min = stats_lookup[match]["Return_Min"]
            median_cost = stats_lookup[match]["Cost_Median"]
        else:
            r_max, r_min, median_cost = 1000.0, 0.0, 10.0

        norm_reward = ((row["Raw_Eval_Reward"] - r_min) / (r_max - r_min + EPSILON)) * 100

        record = row.to_dict()
        record["Clean_Task"] = task_name
        record["Norm_Reward"] = norm_reward
        record["Dataset_Median_Cost"] = median_cost
        processed_records.append(record)

    return pd.DataFrame(processed_records)


# =====================================================================
# 2. MODEL LOADERS (CCDT & VANILLA)
# =====================================================================
def load_ccdt_model(config_path):
    """Initializes and returns a Contrastive CDT model and trainer."""

    def construct_yaml_tuple(loader, node):
        return tuple(loader.construct_sequence(node))

    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", construct_yaml_tuple)

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

    project_name = cfg.get("project", "")
    encoder_type = cfg.get("encoder_type", "front").lower()
    is_back_encoder = "back" in project_name.lower() or encoder_type == "back"
    ModelClass = ContrastiveCDTBack if is_back_encoder else ContrastiveCDTFront
    arch_label = "Back" if is_back_encoder else "Front"
    num_buckets = cfg.get("num_buckets", 1)

    contrastive_weight = cfg.get("contrastive_weight", 0.0)
    if contrastive_weight == 0.0 and "args" in cfg and isinstance(cfg["args"], dict):
        contrastive_weight = cfg["args"].get("contrastive_weight", 0.0)

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
        add_cost_feat=cfg.get("add_cost_feat", False),
        mul_cost_feat=cfg.get("mul_cost_feat", False),
        cat_cost_feat=cfg.get("cat_cost_feat", False),
        action_head_layers=cfg.get("action_head_layers", 1),
        cost_prefix=cfg.get("cost_prefix", False),
    )

    state_dict = model_weights.get("model_state", model_weights.get("model", model_weights))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    trainer = ContrastiveCDTTrainer(
        model, env, cost_boundaries=None, device=DEVICE, reward_scale=cfg["reward_scale"], cost_scale=cfg["cost_scale"]
    )

    clean_task_name = cfg["task"].replace("Offline", "").replace("-v0", "")
    return trainer, cfg, arch_label, num_buckets, float(contrastive_weight), clean_task_name


def load_vanilla_model(config_path):
    """Initializes and returns a native Vanilla CDT baseline model and trainer."""

    def construct_yaml_tuple(loader, node):
        return tuple(loader.construct_sequence(node))

    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", construct_yaml_tuple)

    exp_dir = os.path.dirname(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    checkpoint_dir = os.path.join(exp_dir, "checkpoint")
    model_path = os.path.join(checkpoint_dir, "model_best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoint_dir, "model.pt")

    model_weights = torch.load(model_path, map_location=torch.device(DEVICE))
    seed_all(cfg["seed"])

    base_env = gym.make(cfg["task"])
    env = wrap_env(env=base_env, reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)

    if "cost_limit" in cfg:
        env.set_target_cost(cfg["cost_limit"])

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

    trainer = CDTTrainer(
        model=model,
        env=env,
        reward_scale=cfg["reward_scale"],
        cost_scale=cfg["cost_scale"],
        cost_reverse=cfg.get("cost_reverse", False),
        device=DEVICE,
    )

    clean_task_name = cfg["task"].replace("Offline", "").replace("-v0", "")
    return trainer, cfg, "Vanilla", "Baseline", 0.0, clean_task_name


# =====================================================================
# 3. EVALUATION ENGINE (HANDLES BOTH MODELS)
# =====================================================================
def run_evaluation(log_filter, eval_mode="cost_sweep", model_type="ccdt"):
    stats_lookup = load_dataset_stats()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # Organize runs cleanly by mode and model architecture
    RUN_DIR = os.path.join(BASE_EVAL_DIR, f"{model_type}_{eval_mode}_{timestamp}")
    os.makedirs(RUN_DIR, exist_ok=True)
    output_csv = os.path.join(RUN_DIR, f"raw_data.csv")

    root_dir = LOG_ROOT.get(model_type, LOG_ROOT["ccdt"])
    search_pattern = os.path.join(root_dir, log_filter, "**", "config.yaml")
    config_files = glob.glob(search_pattern, recursive=True)

    if not config_files:
        print(f"❌ No {model_type.upper()} models found for pattern: {search_pattern}")
        return

    print(f"🔍 Executing {eval_mode.upper()} on {len(config_files)} {model_type.upper()} models. Saving to: {RUN_DIR}")

    for config_path in config_files:
        try:
            # Route to the correct model loader
            if model_type == "vanilla":
                trainer, cfg, arch, buckets, cw, task_name = load_vanilla_model(config_path)
            else:
                trainer, cfg, arch, buckets, cw, task_name = load_ccdt_model(config_path)

            match = next((k for k in stats_lookup.keys() if task_name in k), None)
            dataset_max_reward = stats_lookup[match]["Return_Max"] if match else 1000.0

            reward_targets = (
                [dataset_max_reward]
                if eval_mode == "cost_sweep"
                else [dataset_max_reward * m for m in TARGET_REWARD_MULTIPLIERS]
            )

            for t_rew in reward_targets:
                for t_cost in TARGET_COST_SWEEP:
                    print(f"  🚀 Eval | Arch: {arch}-{buckets}B | cw: {cw} | Cost: {t_cost} | Rew: {t_rew:.1f}")

                    raw_ret, raw_cost, ep_len = trainer.evaluate(
                        num_rollouts=NUM_EPISODES,
                        target_return=t_rew * cfg["reward_scale"],
                        target_cost=t_cost * cfg["cost_scale"],
                    )

                    row_data = {
                        "Task": task_name,
                        "Seed": cfg["seed"],
                        "Architecture": arch,
                        "Buckets": buckets,
                        "Contrastive_Weight": cw,
                        "Variant": "Vanilla Baseline" if model_type == "vanilla" else f"{arch}-{buckets}B",
                        "Target_Cost": t_cost,
                        "Target_Reward": t_rew,
                        "Raw_Eval_Cost": raw_cost,
                        "Raw_Eval_Reward": raw_ret,
                        "Avg_Episode_Length": ep_len,
                    }

                    write_header = not os.path.exists(output_csv)
                    pd.DataFrame([row_data]).to_csv(output_csv, mode="a", header=write_header, index=False)
        except Exception as e:
            print(f"❌ Error on {config_path}:")
            traceback.print_exc()

    print(f"\n✅ {model_type.upper()} {eval_mode.upper()} complete. Output: {output_csv}")


# =====================================================================
# 4. PLOTTING ENGINES
# =====================================================================
def append_vanilla_baseline(df, vanilla_csv, stats_lookup):
    if vanilla_csv and os.path.exists(vanilla_csv):
        vanilla_df = load_and_normalize_data(vanilla_csv, stats_lookup)
        vanilla_df["Architecture"] = "Vanilla"
        vanilla_df["Buckets"] = "Baseline"
        vanilla_df["Variant"] = "Vanilla Baseline"
        return pd.concat([df, vanilla_df], ignore_index=True)
    return df


def plot_cost_adherence(run_dir, vanilla_csv=None):
    raw_csv = os.path.join(run_dir, "raw_data.csv")
    stats_lookup = load_dataset_stats()
    df = load_and_normalize_data(raw_csv, stats_lookup)
    df = append_vanilla_baseline(df, vanilla_csv, stats_lookup)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    df["Buckets"] = df["Buckets"].apply(lambda x: f"{x} Buckets" if str(x).isdigit() else str(x))

    tasks = sorted(df["Clean_Task"].unique())
    num_tasks = len(tasks)
    fig, axes = plt.subplots(2, num_tasks, figsize=(5 * num_tasks, 8), sharex=True)
    if num_tasks == 1:
        axes = axes.reshape(2, 1)

    palette = {"Front": "#e74c3c", "Back": "#3498db", "Vanilla": "#2c3e50"}

    for i, task in enumerate(tasks):
        task_df = df[df["Clean_Task"] == task]
        median_ds_cost = task_df["Dataset_Median_Cost"].iloc[0]

        # Row 1: Reward
        sns.lineplot(
            ax=axes[0, i],
            data=task_df,
            x="Target_Cost",
            y="Norm_Reward",
            hue="Architecture",
            style="Buckets",
            palette=palette,
            markers=True,
            dashes=True,
            legend=(i == num_tasks - 1),
        )
        axes[0, i].set_title(task, fontweight="bold")
        axes[0, i].set_ylabel("Normalized Reward (%)" if i == 0 else "")
        axes[0, i].axvline(x=median_ds_cost, color="k", linestyle="--", alpha=0.3)

        # Row 2: Cost
        sns.lineplot(
            ax=axes[1, i],
            data=task_df,
            x="Target_Cost",
            y="Raw_Eval_Cost",
            hue="Architecture",
            style="Buckets",
            palette=palette,
            markers=True,
            dashes=True,
            legend=False,
        )
        sweep_vals = sorted(task_df["Target_Cost"].unique())
        axes[1, i].plot(sweep_vals, sweep_vals, "k:", alpha=0.6, label="Ideal")
        axes[1, i].set_ylabel("Actual Evaluated Cost" if i == 0 else "")
        axes[1, i].set_xlabel("Target Cost Prompt")
        axes[1, i].axvline(x=median_ds_cost, color="k", linestyle="--", alpha=0.3)

    if num_tasks > 0:
        axes[0, num_tasks - 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    plt.tight_layout()
    out_path = os.path.join(run_dir, "plot_adherence_grid.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Adherence plot saved to {out_path}")


def plot_pareto_frontier(run_dir, vanilla_csv=None):
    raw_csv = os.path.join(run_dir, "raw_data.csv")
    stats_lookup = load_dataset_stats()
    df = load_and_normalize_data(raw_csv, stats_lookup)
    df = append_vanilla_baseline(df, vanilla_csv, stats_lookup)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    df["Buckets"] = df["Buckets"].apply(lambda x: f"{x} Buckets" if str(x).isdigit() else str(x))
    df["Variant_Group"] = df["Architecture"] + " - " + df["Buckets"].astype(str)

    tasks = sorted(df["Clean_Task"].unique())
    num_tasks = len(tasks)
    fig, axes = plt.subplots(1, num_tasks, figsize=(6 * num_tasks, 5))
    if num_tasks == 1:
        axes = [axes]

    palette = sns.color_palette("tab10", len(df["Variant_Group"].unique()))

    for i, task in enumerate(tasks):
        task_df = df[df["Clean_Task"] == task]
        ax = axes[i]

        # Scatter all evaluation points to show density
        sns.scatterplot(
            ax=ax,
            data=task_df,
            x="Raw_Eval_Cost",
            y="Norm_Reward",
            hue="Variant_Group",
            palette=palette,
            alpha=0.4,
            legend=False,
        )

        # Calculate and plot the Pareto Frontier line for each variant
        for idx, variant in enumerate(task_df["Variant_Group"].unique()):
            v_df = task_df[task_df["Variant_Group"] == variant].sort_values("Raw_Eval_Cost")
            v_df["Pareto_Reward"] = v_df["Norm_Reward"].cummax()

            sns.lineplot(
                ax=ax,
                data=v_df,
                x="Raw_Eval_Cost",
                y="Pareto_Reward",
                color=palette[idx],
                linewidth=2.5,
                label=variant,
                errorbar=None,
            )

        ax.set_title(f"{task} (Pareto Frontier)", fontweight="bold")
        ax.set_xlabel("Evaluated Cost (Lower is Safer)")
        ax.set_ylabel("Normalized Reward (%) (Higher is Better)" if i == 0 else "")

        if i == num_tasks - 1:
            ax.legend(title="Model Variant", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.get_legend().remove()

    plt.tight_layout()
    out_path = os.path.join(run_dir, "plot_pareto_frontier.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Pareto frontier plot saved to {out_path}")


# =====================================================================
# CLI ROUTER
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Safe RL Eval & Plotting Suite")
    parser.add_argument(
        "mode",
        choices=["eval_cost", "eval_pareto", "plot_cost", "plot_pareto"],
        help="Select the operational module to execute.",
    )

    # Eval arguments
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["ccdt", "vanilla"],
        default="ccdt",
        help="Type of model to evaluate (determines architecture loader and root folder).",
    )
    parser.add_argument(
        "--log_filter",
        type=str,
        default="*_cw*",
        help="Folder pattern for models (e.g., '*_cw04*' for CCDT or 'Vanilla_CDT*' for Vanilla).",
    )

    # Plot arguments
    parser.add_argument("--run_dir", type=str, help="Path to the timestamped run folder (used in plot modes)")
    parser.add_argument(
        "--vanilla_csv", type=str, default=None, help="Path to a generated vanilla baseline CSV to include in plotting"
    )

    args = parser.parse_args()

    if args.mode == "eval_cost":
        run_evaluation(args.log_filter, eval_mode="cost_sweep", model_type=args.model_type)
    elif args.mode == "eval_pareto":
        run_evaluation(args.log_filter, eval_mode="pareto_matrix", model_type=args.model_type)
    elif args.mode == "plot_cost":
        if not args.run_dir:
            raise ValueError("--run_dir is required for plotting.")
        plot_cost_adherence(args.run_dir, args.vanilla_csv)
    elif args.mode == "plot_pareto":
        if not args.run_dir:
            raise ValueError("--run_dir is required for plotting.")
        plot_pareto_frontier(args.run_dir, args.vanilla_csv)
