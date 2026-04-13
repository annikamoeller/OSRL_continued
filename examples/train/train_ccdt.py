import os
import types
from dataclasses import asdict
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import datetime
from torch.utils.data import DataLoader
from tqdm.auto import trange

import bullet_safety_gym  # noqa
import dsrl
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env 
from fsrl.utils import WandbLogger
from osrl.common import SequenceDataset
from osrl.common.exp_util import auto_name, seed_all
from examples.configs.ccdt_configs import ContrastiveCDTTrainConfig, CCDT_DEFAULT_CONFIG
from osrl.algorithms.ccdt import ContrastiveCDTFront, ContrastiveCDTBack, ContrastiveCDTTrainer
from osrl.common.probe_and_vis import evaluate_representations

def get_cost_boundaries(data: dict, num_buckets: int) -> list:
    if num_buckets <= 1: return []
    terminals, timeouts, costs = data['terminals'], data['timeouts'], data['costs']
    episode_ends = np.where(np.logical_or(terminals, timeouts))[0]
    
    episode_costs = []
    start_idx = 0
    for end_idx in episode_ends:
        episode_costs.append(np.sum(costs[start_idx:end_idx + 1]))
        start_idx = end_idx + 1
        
    quantiles = np.linspace(0, 1, num_buckets + 1)[1:-1]
    boundaries = np.quantile(episode_costs, quantiles).tolist()
    print(f"\n🚀 [CCDT Setup] Global Cost Boundaries for {num_buckets} buckets: {boundaries}\n")
    return boundaries

@pyrallis.wrap()
def train(args: ContrastiveCDTTrainConfig):
    # 1. Clean Config Merging (Safely preserving Thesis variables)
    parsed_cfg = asdict(args)
    default_cfg = asdict(ContrastiveCDTTrainConfig())
    
    # Find what you explicitly typed in the terminal
    cli_overrides = {k: parsed_cfg[k] for k in parsed_cfg.keys() if parsed_cfg[k] != default_cfg[k]}
    
    # Start with all your parsed args (preserves pretrain_steps, num_buckets, etc.)
    final_cfg = asdict(args)
    # Safely pull in the vanilla tuned hyperparams for this specific task (e.g., reward_scale)
    final_cfg.update(asdict(CCDT_DEFAULT_CONFIG[args.task]()))
    # Re-apply your terminal overrides so they get the final say
    final_cfg.update(cli_overrides)
    cfg = final_cfg
    args = types.SimpleNamespace(**cfg)
    
    args = types.SimpleNamespace(**final_cfg)

    # 2. Thesis Naming Convention
    env_short = args.task.split("-")[0].replace("Offline", "")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")    
    args.name = f"{env_short}_{args.num_buckets}B_{args.pretrain_steps}Pre_{timestamp}"
    if args.group is None:
        args.group = f"{args.task}-contrastive-experiments"
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
        os.makedirs(args.logdir, exist_ok=True)
        
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)
    seed_all(args.seed)

    # 3. Environment & Wrapper Setup (RESTORED)
    if "Metadrive" in args.task:
        import gym as old_gym
        env = old_gym.make(args.task)
    else:
        env = gym.make(args.task)

    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)
    cost_boundaries = get_cost_boundaries(data, args.num_buckets)

    # Restore data pre-processing filters
    cbins, rbins, max_npb, min_npb = None, None, None, None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins, rbins = density_cfg["cbins"], density_cfg["rbins"]
        max_npb, min_npb = density_cfg["max_npb"], density_cfg["min_npb"]
        
    data = env.pre_process_data(data, args.outliers_percent, args.noise_scale,
                                args.inpaint_ranges, args.epsilon, args.density,
                                cbins=cbins, rbins=rbins, max_npb=max_npb, min_npb=min_npb)
    # Restore wrappers
    env = wrap_env(env=env, reward_scale=args.reward_scale)
    env = OfflineEnvWrapper(env)

    # 4. Model Setup
    model_kwargs = dict(
        state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0], embedding_dim=args.embedding_dim,
        contrastive_dim=args.contrastive_dim, seq_len=args.seq_len,
        episode_len=args.episode_len, num_layers=args.num_layers, num_heads=args.num_heads,
        attention_dropout=args.attention_dropout, residual_dropout=args.residual_dropout,
        embedding_dropout=args.embedding_dropout, time_emb=args.time_emb,
        use_rew=args.use_rew, use_cost=args.use_cost, cost_transform=args.cost_transform,
        add_cost_feat=args.add_cost_feat, mul_cost_feat=args.mul_cost_feat,
        cat_cost_feat=args.cat_cost_feat, action_head_layers=args.action_head_layers,
        cost_prefix=args.cost_prefix, stochastic=args.stochastic,
        init_temperature=args.init_temperature, target_entropy=-env.action_space.shape[0],
    )

    # Create either Front or Back Encoder version
    if getattr(args, "encoder_type", "back").lower() == "front":
        print("\n🚀 [CCDT Setup] Initializing FRONT-Encoder Architecture...\n")
        model = ContrastiveCDTFront(**model_kwargs).to(args.device)
    else:
        print("\n🚀 [CCDT Setup] Initializing TRUE BACK-Encoder Architecture...\n")
        model = ContrastiveCDTBack(**model_kwargs).to(args.device)

    def checkpoint_fn():
        return {"model_state": model.state_dict()}
    logger.setup_checkpoint_fn(checkpoint_fn)
    

    def checkpoint_fn():
        return {"model_state": model.state_dict()}
    logger.setup_checkpoint_fn(checkpoint_fn)

    # 5. Trainer Setup
    trainer = ContrastiveCDTTrainer(
        model, env, logger=logger, contrastive_weight=args.contrastive_weight,
        temperature=args.temperature, num_buckets=args.num_buckets,         
        cost_boundaries=cost_boundaries, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, betas=args.betas, clip_grad=args.clip_grad,
        lr_warmup_steps=args.lr_warmup_steps, reward_scale=args.reward_scale,
        cost_scale=args.cost_scale, loss_cost_weight=args.loss_cost_weight,
        loss_state_weight=args.loss_state_weight, cost_reverse=args.cost_reverse,
        no_entropy=args.no_entropy, device=args.device
    )

    # 6. Dataset Setup
    ct = lambda x: 70 - x if args.linear else 1 / (x + 10)
    dataset = SequenceDataset(
        data, seq_len=args.seq_len, reward_scale=args.reward_scale,
        cost_scale=args.cost_scale, deg=args.deg, pf_sample=args.pf_sample,
        max_rew_decrease=args.max_rew_decrease, beta=args.beta,
        augment_percent=args.augment_percent, cost_reverse=args.cost_reverse,
        max_reward=args.max_reward, min_reward=args.min_reward,
        pf_only=args.pf_only, rmin=args.rmin, cost_bins=args.cost_bins,
        npb=args.npb, cost_sample=args.cost_sample, cost_transform=ct,
        start_sampling=args.start_sampling, prob=args.prob,
        random_aug=args.random_aug, aug_rmin=args.aug_rmin,
        aug_rmax=args.aug_rmax, aug_cmin=args.aug_cmin, aug_cmax=args.aug_cmax,
        cgap=args.cgap, rstd=args.rstd, cstd=args.cstd,
    )
    
    trainloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    trainloader_iter = iter(trainloader)

    # 7. Training Loops
    best_reward = -np.inf
    best_cost = np.inf
    best_idx = 0

    if args.pretrain_steps > 0:
        for _ in trange(args.pretrain_steps, desc="Pre-training"):
            try: batch = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                batch = next(trainloader_iter)
            batch = [b.to(args.device) for b in batch]
            trainer.train_one_step(*batch, is_pretraining=True)

    for step in trange(args.update_steps, desc="Training"):
        try: batch = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)
        
        batch = [b.to(args.device) for b in batch]
        trainer.train_one_step(*batch, is_pretraining=False)

        # Probing Integration
        if (step + 1) % args.probe_every == 0:
            evaluate_representations(trainer, trainloader, args.device, step, num_buckets=args.num_buckets)

        # Evaluation & Saving
        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:
            average_reward, average_cost = [], []
            log_cost, log_reward, log_len = {}, {}, {}
            
            for target_return in args.target_returns:
                reward_return, cost_return = target_return
                if args.cost_reverse:
                    ret, cost, length = trainer.evaluate(
                        args.eval_episodes, reward_return * args.reward_scale,
                        (args.episode_len - cost_return) * args.cost_scale)
                else:
                    ret, cost, length = trainer.evaluate(
                        args.eval_episodes, reward_return * args.reward_scale,
                        cost_return * args.cost_scale)
                average_cost.append(cost)
                average_reward.append(ret)

                name = f"c_{int(cost_return)}_r_{int(reward_return)}"
                log_cost.update({name: cost})
                log_reward.update({name: ret})
                log_len.update({name: length})

            logger.store(tab="cost", **log_cost)
            logger.store(tab="ret", **log_reward)
            logger.store(tab="length", **log_len)

            logger.save_checkpoint()
            mean_ret, mean_cost = np.mean(average_reward), np.mean(average_cost)
            if mean_cost < best_cost or (mean_cost == best_cost and mean_ret > best_reward):
                best_cost, best_reward, best_idx = mean_cost, mean_ret, step
                logger.save_checkpoint(suffix="best")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)
        else:
            logger.write_without_reset(step)

if __name__ == "__main__":
    train()