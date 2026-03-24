import gymnasium as gym
import pyrallis
import torch
from dataclasses import dataclass
from typing import List
from pyrallis import field

from osrl.algorithms.ccdt import ContrastiveCDT, ContrastiveCDTTrainer
from osrl.common.exp_util import load_config_and_model, seed_all
from dsrl.offline_env import OfflineEnvWrapper, wrap_env

@dataclass
class EvalConfig:
    path: str = "logs/path_to_your_model/model.pt"
    # Defaults to the "Expert" levels usually found in the paper
    returns: List[float] = field(default=[400, 500, 600], is_mutable=True)
    costs: List[float] = field(default=[10, 10, 10], is_mutable=True)
    eval_episodes: int = 10
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

@pyrallis.wrap()
def eval(args: EvalConfig):
    # 1. Load the saved config and weights
    cfg, model_state = load_config_and_model(args.path)
    seed_all(cfg["seed"])

    # 2. Setup the physical environment
    env = wrap_env(gym.make(cfg["task"]), reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)

    # 4. Initialize the ContrastiveCDT exactly as it was trained
    model = ContrastiveCDT(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=cfg["embedding_dim"],
        contrastive_dim=cfg.get("contrastive_dim", 64), # Graceful fallback
        seq_len=cfg["seq_len"],
        episode_len=cfg["episode_len"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        attention_dropout=cfg["attention_dropout"],
        residual_dropout=cfg["residual_dropout"],
        embedding_dropout=cfg["embedding_dropout"],
        time_emb=cfg["time_emb"],
        use_rew=cfg["use_rew"],
        use_cost=cfg["use_cost"],
        cost_transform=cfg["cost_transform"],
        add_cost_feat=cfg["add_cost_feat"],
        mul_cost_feat=cfg["mul_cost_feat"],
        cat_cost_feat=cfg["cat_cost_feat"],
        action_head_layers=cfg["action_head_layers"],
        cost_prefix=cfg["cost_prefix"],
        stochastic=cfg["stochastic"],
        init_temperature=cfg["init_temperature"],
        target_entropy=-env.action_space.shape[0],
    )
    
    model.load_state_dict(model_state["model_state"])
    model.to(args.device)

    # 4. Use the Trainer to run rollouts
    trainer = ContrastiveCDTTrainer(
        model, env, device=args.device,
        reward_scale=cfg["reward_scale"],
        cost_scale=cfg["cost_scale"]
    )

    for target_ret, target_cost in zip(args.returns, args.costs):
        ret, cost, length = trainer.evaluate(args.eval_episodes, 
                                             target_ret * cfg["reward_scale"], 
                                             target_cost * cfg["cost_scale"])
        print(f"Prompt: R={target_ret} C={target_cost} | Result: R={ret:.2f} C={cost:.2f} Len={length}")

if __name__ == "__main__":
    eval()