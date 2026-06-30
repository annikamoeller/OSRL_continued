#!/usr/bin/env python3
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import yaml
from tqdm import tqdm
import bullet_safety_gym  # noqa
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.algorithms.ccdt import ContrastiveCDTBack
from osrl.algorithms.cdt import CDT
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

# --- PATH SETUP ---
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Your provided dictionary
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

# Empirical Max Rewards for scaling
TARGET_REWARDS = {"AntRun": 950.0, "CarCircle": 500.0, "DroneRun": 650.0}

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

def run_eval(model, env, cfg, target_return, num_episodes=100):
    results = {'rewards': [], 'costs': []}
    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        state, _ = env.reset()
        ep_ret, ep_cost = 0, 0
        # Decision Transformer Context
        states = torch.zeros((1, cfg["episode_len"]+1, env.observation_space.shape[0]), device=DEVICE)
        actions = torch.zeros((1, cfg["episode_len"], env.action_space.shape[0]), device=DEVICE)
        rewards = torch.zeros((1, cfg["episode_len"]), device=DEVICE)
        costs = torch.zeros((1, cfg["episode_len"]), device=DEVICE)
        
        states[0, 0] = torch.tensor(state, device=DEVICE)
        target_cost = 0.0 # Min cost target
        
        for t in range(cfg["episode_len"]):
            # Prompting
            rewards[0, t] = target_return * cfg["reward_scale"]
            costs[0, t] = target_cost * cfg["cost_scale"]
            
            # Inference
            with torch.no_grad():
                # We assume model provides .get_action or standard forward
                action_preds, cost_preds, state_preds = model(
                states[:, :t+1],
                actions[:, :t+1],
                rewards[:, :t+1],
                costs[:, :t+1],
                torch.arange(t+1, device=DEVICE).unsqueeze(0)
            )
                action_preds=action_preds.mean
                # 2. Extract only the last action (the one for the current state)
                action = action_preds[0, -1]
            
            action_np = action.cpu().numpy()
            state, reward, term, trunc, info = env.step(action_np)
            
            ep_ret += reward
            ep_cost += info.get('cost', 0.0)
            
            if term or trunc: break
            states[0, t+1] = torch.tensor(state, device=DEVICE)
            actions[0, t] = torch.tensor(action_np, device=DEVICE)
            
        results['rewards'].append(ep_ret)
        results['costs'].append(ep_cost)
    return results

# MAIN LOOP
all_results = []
for env_name, models in EXPERIMENTS.items():
    for model_name, path in models.items():
        print(f"Processing {env_name} - {model_name}...")
        model, env, cfg = load_and_hook_model(path, is_vanilla=("Baseline" in model_name))
        
        res = run_eval(model, env, cfg, TARGET_REWARDS[env_name])
        for r, c in zip(res['rewards'], res['costs']):
            all_results.append({'Env': env_name, 'Model': model_name, 'Reward': r, 'Cost': c})

# PLOTTING
df = pd.DataFrame(all_results)
g = sns.FacetGrid(df, col="Env", row="Metric", sharex=False, sharey=False, margin_titles=True)
# Melt to separate metrics
df_melted = df.melt(id_vars=['Env', 'Model'], value_vars=['Reward', 'Cost'], var_name='Metric', value_name='Value')

plt.figure(figsize=(15, 8))
g = sns.FacetGrid(df_melted, col="Env", row="Metric", hue="Model", sharex=False, sharey=False)
g.map_dataframe(sns.histplot, "Value", element="step", fill=False, kde=True)
g.add_legend()
plt.savefig("performance_distribution.png", dpi=300)