import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ==============================================================================
# 1. EVALUATION ROLLOUT WRAPPER
# ==============================================================================

def get_ood_trajectories(model, env, cfg, max_reward, num_episodes=100):
    """
    Runs the model with the OOD Prompt: Absolute zero cost, maximum reward.
    Returns the raw cost and return arrays for the 2D distribution plot.
    """
    target_ret = max_reward * cfg.get("reward_scale", 1.0)
    target_cost = 0.0  # STRICT ZERO COST PROMPT
    
    total_costs, total_rets = [], []
    device = next(model.parameters()).device
    
    print(f"Rolling out {num_episodes} OOD episodes [Target Cost: 0.0, Target Ret: {max_reward}]...")
    
    with torch.no_grad():
        for _ in range(num_episodes):
            state, _ = env.reset()
            ep_cost, ep_ret = 0.0, 0.0

            # Initialize sliding window context
            states = torch.zeros((1, cfg["seq_len"], env.observation_space.shape[0]), device=device)
            actions = torch.zeros((1, cfg["seq_len"], env.action_space.shape[0]), device=device)
            rewards = torch.zeros((1, cfg["seq_len"]), device=device)
            costs = torch.zeros((1, cfg["seq_len"]), device=device)
            time_steps = torch.zeros((1, cfg["seq_len"]), dtype=torch.long, device=device)

            ret_to_go = target_ret
            cost_to_go = target_cost

            for t in range(cfg["episode_len"]):
                states[0, -1] = torch.tensor(state, dtype=torch.float32, device=device)
                rewards[0, -1] = ret_to_go
                costs[0, -1] = cost_to_go
                time_steps[0, -1] = t

                # Get action from the transformer
                action_tensor = model(states, actions, rewards, costs, time_steps)
                action = action_tensor.squeeze().cpu().numpy()
                
                next_state, reward, terminated, truncated, info = env.step(action)

                cost = info.get("cost", 0.0)
                ep_cost += cost
                ep_ret += reward

                # Shift window
                states = torch.cat([states[:, 1:], torch.zeros((1, 1, states.shape[-1]), device=device)], dim=1)
                actions[0, -1] = torch.tensor(action, dtype=torch.float32, device=device)
                actions = torch.cat([actions[:, 1:], torch.zeros((1, 1, actions.shape[-1]), device=device)], dim=1)
                rewards = torch.cat([rewards[:, 1:], torch.zeros((1, 1), device=device)], dim=1)
                costs = torch.cat([costs[:, 1:], torch.zeros((1, 1), device=device)], dim=1)
                time_steps = torch.cat([time_steps[:, 1:], torch.zeros((1, 1), dtype=torch.long, device=device)], dim=1)

                ret_to_go -= reward * cfg.get("reward_scale", 1.0)
                cost_to_go -= cost * cfg.get("cost_scale", 1.0)
                state = next_state

                if terminated or truncated:
                    break

            total_costs.append(ep_cost)
            total_rets.append(ep_ret)
            
    return np.array(total_costs), np.array(total_rets)

# ==============================================================================
# 2. PLOTTING FUNCTION
# ==============================================================================

def plot_bimodal_kde(cdt_costs, cdt_returns, ccdt_costs, ccdt_returns, env_name, target_reward, save_dir="plots"):
    """
    Generates the 2D Kernel Density Estimate (KDE) distribution plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Plot Vanilla CDT Baseline (Red)
    sns.kdeplot(
        x=cdt_costs, y=cdt_returns, 
        cmap="Reds", fill=True, alpha=0.5, thresh=0.05, 
        label="Vanilla CDT", ax=ax
    )
    
    # 2. Plot CCDT (Blue)
    sns.kdeplot(
        x=ccdt_costs, y=ccdt_returns, 
        cmap="Blues", fill=True, alpha=0.6, thresh=0.05, 
        label="CCDT", ax=ax
    )

    # 3. Plot the OOD Golden Target
    ax.scatter(
        0.0, target_reward, 
        color='gold', marker='*', s=400, edgecolors='black', zorder=5
    )

    # 4. Reference Lines
    ax.axvline(x=10.0, color='red', linestyle='--', linewidth=2, alpha=0.8)

    # Formatting
    ax.set_title(f"OOD Target Generalization: {env_name}\n(Prompted: Min Cost, Max Reward)", fontsize=15, weight='bold')
    ax.set_xlabel("Cumulative Trajectory Cost (\u2193 Lower is Better)", fontsize=13)
    ax.set_ylabel("Cumulative Trajectory Reward (\u2191 Higher is Better)", fontsize=13)
    
    # Dynamic axis limits
    max_c = max(np.max(cdt_costs), np.max(ccdt_costs)) if len(cdt_costs) else 20.0
    min_r = min(np.min(cdt_returns), np.min(ccdt_returns)) if len(cdt_returns) else 0.0
    ax.set_xlim(-2.0, max(max_c * 1.1, 15.0))
    ax.set_ylim(min_r * 0.9, target_reward * 1.1)

    # Custom Legend
    legend_elements = [
        Patch(facecolor='red', alpha=0.5, label='Vanilla CDT Distribution'),
        Patch(facecolor='blue', alpha=0.6, label='CCDT Distribution'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=15, markeredgecolor='black', label='OOD Prompt Target'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Safety Limit (10.0)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{env_name}_ood_kde_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"\u2728 Plot successfully generated: {save_path}")
    plt.close()

# ==============================================================================
# 3. EXECUTION ENTRYPOINT
# ==============================================================================

def run_ood_analysis(cdt_model, ccdt_model, env, env_name, cfg, max_dataset_reward):
    """
    Main entrypoint. Pass your instantiated models, environment, and config here.
    """
    print(f"--- Starting OOD Analysis for {env_name} ---")
    
    # 1. Gather CDT Data
    print("\nRunning Vanilla CDT Baseline...")
    cdt_costs, cdt_returns = get_ood_trajectories(cdt_model, env, cfg, max_reward=max_dataset_reward)
    
    # 2. Gather CCDT Data
    print("\nRunning CCDT...")
    ccdt_costs, ccdt_returns = get_ood_trajectories(ccdt_model, env, cfg, max_reward=max_dataset_reward)
    
    # 3. Generate the Visualization
    print("\nGenerating 2D Bimodal KDE Plot...")
    plot_bimodal_kde(cdt_costs, cdt_returns, ccdt_costs, ccdt_returns, env_name, target_reward=max_dataset_reward)

if __name__ == "__main__":
    # --- USAGE EXAMPLE ---
    cdt_model = load_my_cdt()
    ccdt_model = load_my_ccdt()
    env = gym.make("OfflineAntRun-v0")
    cfg = {"seq_len": 20, "episode_len": 1000, "reward_scale": 1.0, "cost_scale": 1.0}
    
    run_ood_analysis(cdt_model, ccdt_model, env, "OfflineAntRun-v0", cfg, max_dataset_reward=150.0)
    pass