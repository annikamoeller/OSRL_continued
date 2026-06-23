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
from osrl.common.exp_util import load_config_and_model, seed_all
from osrl.algorithms.ccdt import ContrastiveCDTBack
from osrl.algorithms.cdt import CDT

# --- CONFIGURATION (UPDATE THESE PATHS TO YOUR BEST MODELS) ---
# Pick ONE specific AntRun environment model for both to make it a fair comparison
VANILLA_CONFIG_PATH = os.path.join(PROJECT_ROOT, "output_cdt/Vanilla_CDT_OfflineAntRun-v0/checkpoint/config.yaml")
CCDT_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "thesis_final_models/Back_OfflineAntRun-v0_2Buckets_cw04/checkpoint/config.yaml"
)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1. THE WIRETAP (PYTORCH HOOK)
# We use a global list to store the hidden states as the model thinks.
hidden_states_buffer = []


def hidden_state_hook(module, inp, out):
    """Intercepts the input to the action head (the final Transformer representation)."""
    hidden = inp[0].detach().cpu()
    # Sequence models output [Batch, Seq_Len, Hidden_Dim]. We only want the current timestep (-1).
    if hidden.dim() == 3:
        hidden = hidden[:, -1, :]
    elif hidden.dim() == 2:
        pass  # Already flat
    hidden_states_buffer.append(hidden.numpy())


# 2. MODEL LOADING & HOOK ATTACHMENT
def load_and_hook_model(config_path, is_vanilla=True):
    exp_dir = os.path.dirname(os.path.dirname(config_path))
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load weights
    try:
        _, model_weights = load_config_and_model(exp_dir, best=False)
    except:
        _, model_weights = load_config_and_model(exp_dir, best=True)

    base_env = gym.make(cfg["task"])
    env = wrap_env(env=base_env, reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)

    # Init architectures
    if is_vanilla:
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
        )

    state_dict = model_weights.get("model_state", model_weights.get("model", model_weights))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 🚨 ATTACH THE HOOK 🚨
    # We target the final linear layer that predicts the action.
    # The exact name depends on the OSRL implementation, usually `predict_action` or `action_head`
    hook_handle = None
    for name, module in model.named_modules():
        if "predict_action" in name or "action_head" in name:
            hook_handle = module.register_forward_hook(hidden_state_hook)
            print(f"  📎 Hook successfully attached to: {name}")
            break

    if hook_handle is None:
        print("  ⚠️ Warning: Could not find action head to hook. Falling back to base module.")
        model.register_forward_hook(hidden_state_hook)

    return model, env, cfg


# 3. DATA COLLECTION LOOP
def collect_embeddings(model, env, cfg, target_cost_list, num_steps=200):
    global hidden_states_buffer
    hidden_states_buffer = []  # Clear buffer
    labels = []

    # We will run the model manually to control exactly when we capture data
    for t_cost in target_cost_list:
        state, info = env.reset()

        # Initialize context buffers
        states = torch.zeros(
            (1, cfg["episode_len"] + 1, env.observation_space.shape[0]), dtype=torch.float32, device=DEVICE
        )
        actions = torch.zeros((1, cfg["episode_len"], env.action_space.shape[0]), dtype=torch.float32, device=DEVICE)
        rewards = torch.zeros((1, cfg["episode_len"], 1), dtype=torch.float32, device=DEVICE)
        costs = torch.zeros((1, cfg["episode_len"], 1), dtype=torch.float32, device=DEVICE)

        ep_return = target_reward = 800.0 * cfg["reward_scale"]  # High reward prompt
        ep_cost = t_cost * cfg["cost_scale"]

        states[0, 0] = torch.tensor(state, device=DEVICE)

        for step in range(num_steps):
            # Pad sequences for the transformer
            seq_start = max(0, step + 1 - cfg["seq_len"])
            s_seq = states[:, seq_start : step + 1]
            a_seq = actions[:, seq_start : step + 1]
            r_seq = rewards[:, seq_start : step + 1]
            c_seq = costs[:, seq_start : step + 1]

            with torch.no_grad():
                # This forward pass triggers our wiretap hook!
                if hasattr(model, "get_action"):
                    # Call the wrapper if it exists
                    action = model.get_action(s_seq, a_seq, r_seq, c_seq)
                else:
                    # Direct forward pass
                    out = model(s_seq, a_seq, r_seq, c_seq)
                    action = out[0] if isinstance(out, tuple) else out

            # We record whether the model was *told* to be safe or unsafe
            label = "Safe Trajectory (Cost=0)" if t_cost == 0.0 else "Unsafe Trajectory (Cost=80)"
            labels.append(label)

            # Step environment (we don't strictly care about the physical outcome, just the mental state)
            state, reward, terminated, truncated, info = env.step(
                action[0, -1].cpu().numpy() if action.dim() == 3 else action.cpu().numpy()
            )
            if terminated or truncated:
                break

            states[0, step + 1] = torch.tensor(state, device=DEVICE)
            actions[0, step] = torch.tensor(action[0, -1] if action.dim() == 3 else action, device=DEVICE)

    # Flatten the collected buffer
    embeddings = np.concatenate(hidden_states_buffer, axis=0)
    return embeddings, labels


# 4. EXECUTION AND VISUALIZATION
if __name__ == "__main__":
    print("🚀 Extracting representations for Thesis Motivation Plot...")

    # 1. Vanilla Extraction
    print("\n🧠 Processing Vanilla CDT...")
    v_model, v_env, v_cfg = load_and_hook_model(VANILLA_CONFIG_PATH, is_vanilla=True)
    v_embeddings, v_labels = collect_embeddings(v_model, v_env, v_cfg, target_cost_list=[0.0, 80.0])

    # 2. CCDT Extraction
    print("\n🧠 Processing Contrastive CCDT...")
    c_model, c_env, c_cfg = load_and_hook_model(CCDT_CONFIG_PATH, is_vanilla=False)
    c_embeddings, c_labels = collect_embeddings(c_model, c_env, c_cfg, target_cost_list=[0.0, 80.0])

    # 3. Dimensionality Reduction (t-SNE)
    print("\n🌌 Crushing high-dimensional thoughts to 2D using t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")

    v_tsne = tsne.fit_transform(v_embeddings)
    c_tsne = tsne.fit_transform(c_embeddings)

    # Format into dataframes
    df_v = pd.DataFrame({"x": v_tsne[:, 0], "y": v_tsne[:, 1], "Condition": v_labels, "Model": "Vanilla CDT"})
    df_c = pd.DataFrame({"x": c_tsne[:, 0], "y": c_tsne[:, 1], "Condition": c_labels, "Model": "Contrastive CDT"})
    df_plot = pd.concat([df_v, df_c])

    # 4. Plot the results
    print("🎨 Rendering thesis graphics...")
    sns.set_theme(style="white", font_scale=1.2)
    plt.rcParams.update({"font.family": "serif"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)

    palette = {"Safe Trajectory (Cost=0)": "#2ecc71", "Unsafe Trajectory (Cost=80)": "#e74c3c"}

    # Plot Vanilla Hairball
    sns.scatterplot(
        ax=axes[0], data=df_v, x="x", y="y", hue="Condition", palette=palette, s=60, alpha=0.7, edgecolor=None
    )
    axes[0].set_title("Vanilla CDT Latent Space\n(Representational Collapse)", fontweight="bold")
    axes[0].set_xlabel("t-SNE Dim 1")
    axes[0].set_ylabel("t-SNE Dim 2")
    axes[0].get_legend().remove()

    # Plot CCDT Split
    sns.scatterplot(
        ax=axes[1], data=df_c, x="x", y="y", hue="Condition", palette=palette, s=60, alpha=0.7, edgecolor=None
    )
    axes[1].set_title("Contrastive CDT Latent Space\n(Structured Safety Manifold)", fontweight="bold")
    axes[1].set_xlabel("t-SNE Dim 1")
    axes[1].set_ylabel("")
    axes[1].legend(title="Prompted Behavior", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Clean up axes
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#bdc3c7")

    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, "examples/eval/eval_suite/thesis_hairball_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Motivation Plot generated successfully: {out_path}")
