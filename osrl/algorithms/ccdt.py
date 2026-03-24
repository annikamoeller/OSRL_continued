from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_metric_learning.losses import NTXentLoss

# Import the original base classes
from osrl.algorithms.cdt import CDT, CDTTrainer

class ContrastiveCDT(CDT): 
    def __init__(self, contrastive_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_dim = contrastive_dim
        
        # SimCLR-style deep projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, contrastive_dim)
        )

    def get_latents(self, state_emb, action_emb, cost_emb):
        # "Product" mechanism: condition the representation on the cost budget
        state_conditioned = state_emb * cost_emb
        action_conditioned = action_emb * cost_emb
        
        # Combine and project
        s_a_stack = state_conditioned + action_conditioned
        return self.contrastive_head(s_a_stack)

    def forward(
            self,
            states: torch.Tensor,  
            actions: torch.Tensor, 
            returns_to_go: torch.Tensor, 
            costs_to_go: torch.Tensor,  
            time_steps: torch.Tensor,  
            padding_mask: Optional[torch.Tensor] = None, 
            episode_cost: torch.Tensor = None,  
            return_latents: bool = False, # <--- THESIS ADDITION
    ) -> torch.FloatTensor:
        
        # ==========================================================
        # EXACT COPY OF ORIGINAL CDT FORWARD PASS (To maintain stability)
        # ==========================================================
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        if self.time_emb:
            timestep_emb = self.timestep_emb(time_steps)
        else:
            timestep_emb = 0.0
        state_emb = self.state_emb(states) + timestep_emb
        act_emb = self.action_emb(actions) + timestep_emb

        seq_list = [state_emb, act_emb]

        if self.cost_transform is not None:
            costs_to_go = self.cost_transform(costs_to_go.detach())

        if self.use_cost:
            costs_emb = self.cost_emb(costs_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, costs_emb)
        if self.use_rew:
            returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, returns_emb)

        sequence = torch.stack(seq_list, dim=1).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch_size, self.seq_repeat * seq_len, self.embedding_dim)

        if padding_mask is not None:
            padding_mask = torch.stack([padding_mask] * self.seq_repeat, dim=1).permute(0, 2, 1).reshape(batch_size, -1)

        if self.cost_prefix:
            episode_cost_expanded = episode_cost.unsqueeze(-1).unsqueeze(-1).to(states.dtype)
            episode_cost_emb = self.prefix_emb(episode_cost_expanded)
            sequence = torch.cat([episode_cost_emb, sequence], dim=1)
            if padding_mask is not None:
                padding_mask = torch.cat([padding_mask[:, :1], padding_mask], dim=1)

        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        if self.cost_prefix:
            out = out[:, 1:]

        out = out.reshape(batch_size, seq_len, self.seq_repeat, self.embedding_dim)
        out = out.permute(0, 2, 1, 3)

        action_feature = out[:, self.seq_repeat - 1]
        state_feat = out[:, self.seq_repeat - 2]

        if self.add_cost_feat and self.use_cost:
            state_feat = state_feat + costs_emb.detach()
        if self.mul_cost_feat and self.use_cost:
            state_feat = state_feat * costs_emb.detach()
        if self.cat_cost_feat and self.use_cost:
            state_feat = torch.cat([state_feat, costs_emb.detach()], dim=2)

        action_preds = self.action_head(state_feat) 
        cost_preds = F.log_softmax(self.cost_pred_head(action_feature), dim=-1)
        state_preds = self.state_pred_head(action_feature) 

        # ==========================================================
        # THESIS ADDITION: LATENT EXTRACTION
        # ==========================================================
        if return_latents and self.use_cost:
            latents = self.get_latents(state_emb, act_emb, costs_emb)
            return action_preds, cost_preds, state_preds, latents
            
        return action_preds, cost_preds, state_preds


class ContrastiveCDTTrainer(CDTTrainer): 
    def __init__(self, model, env, contrastive_weight=0.1, temperature=0.1, 
                 num_buckets=2, cost_boundaries=None, **kwargs):
        # Initialize the parent class (sets up optimizers, schedulers, etc.)
        super().__init__(model, env, **kwargs)
        
        self.contrastive_weight = contrastive_weight
        self.num_buckets = num_buckets
        if cost_boundaries is not None:
            self.cost_boundaries = torch.tensor(cost_boundaries, device=self.device)
        self.ntxent_loss = NTXentLoss(temperature=temperature)
        
    def train_one_step(self, states, actions, returns, costs_return, time_steps, mask, episode_cost, costs, is_pretraining=False):
        padding_mask = ~mask.to(torch.bool)
        
        # 1. Forward Pass (Request Latents from our custom ContrastiveCDT)
        action_preds, cost_preds, state_preds, latents = self.model(
            states=states, actions=actions, returns_to_go=returns,
            costs_to_go=costs_return, time_steps=time_steps, 
            padding_mask=padding_mask, episode_cost=episode_cost, 
            return_latents=True
        )

        # 2. Standard Decision Transformer Losses (Exact logic from parent CDTTrainer)
        if self.stochastic:
            log_likelihood = action_preds.log_prob(actions)[mask > 0].mean()
            entropy = action_preds.entropy()[mask > 0].mean()
            entropy_reg = 0.0 if self.no_entropy else self.model.temperature().detach()
            act_loss = -(log_likelihood + entropy_reg * entropy)
        else:
            act_loss = F.mse_loss(action_preds, actions.detach(), reduction="none")
            act_loss = (act_loss * mask.unsqueeze(-1)).mean()

        cost_preds_flat = cost_preds.reshape(-1, 2)
        costs_flat = costs.flatten().long().detach()
        cost_loss = (F.nll_loss(cost_preds_flat, costs_flat, reduction="none") * mask.flatten()).mean()

        state_loss = F.mse_loss(state_preds[:, :-1], states[:, 1:].detach(), reduction="none")
        state_loss = (state_loss * mask[:, :-1].unsqueeze(-1)).mean()

        # 3. Contrastive Task: Trajectory-Aware Bucketing
        if self.cost_boundaries is not None: # cont loss = 0 if no cost boundaries given (contrastive loss turned off)
            flat_latents = latents[mask > 0]
        
            # Crush episode_cost to 1D and bucketize based on the dynamic boundaries
            ep_cost_1d = episode_cost.view(-1)
            traj_labels = torch.bucketize(ep_cost_1d, self.cost_boundaries).long()

            # Broadcast labels to match sequence length and flatten
            seq_len = states.shape[1]
            batch_labels = traj_labels.unsqueeze(1).expand(-1, seq_len)
            flat_labels = batch_labels[mask > 0].long()
            
            # --- Anti-OOM Shield --- (Limit to 128 elements for Contrastive Loss)
            MAX_SAMPLES = 128
            if flat_latents.shape[0] > MAX_SAMPLES:
                idx = torch.randperm(flat_latents.shape[0], device=self.device)[:MAX_SAMPLES]
                flat_latents = flat_latents[idx]
                flat_labels = flat_labels[idx]

            if torch.isnan(flat_latents).any():
                flat_latents = torch.nan_to_num(flat_latents)
            cont_loss = self.ntxent_loss(flat_latents, flat_labels)
            
        else:
            cont_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
        # 4. Combined Optimization Strategy
        if is_pretraining:
            loss = cont_loss 
        else:
            loss = (act_loss + 
                    self.cost_weight * cost_loss + 
                    self.state_weight * state_loss + 
                    self.contrastive_weight * cont_loss)            
            
        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optim.step()
        
        # Temperature update for stochastic actions
        if self.stochastic:
            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (self.model.temperature() * (entropy - self.model.target_entropy).detach())
            temperature_loss.backward()
            self.log_temperature_optimizer.step()
            
        self.scheduler.step()

        # 5. Log exactly as OSRL does
        self.logger.store(
            tab="train",
            total_loss=loss.item(),
            cont_loss=cont_loss.item(),
            act_loss=act_loss.item() if not is_pretraining else 0.0,
            cost_loss=cost_loss.item() if not is_pretraining else 0.0,
            state_loss=state_loss.item() if not is_pretraining else 0.0,
            train_lr=self.scheduler.get_last_lr()[0],
            num_buckets=len(torch.unique(flat_labels))
        )