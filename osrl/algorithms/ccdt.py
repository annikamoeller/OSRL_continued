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

import torch
import torch.nn as nn
import torch.nn.functional as F
from osrl.algorithms.cdt import CDT

class BaseContrastiveCDT(CDT):
    """
    Shared base class that holds the heavy lifting helper functions 
    so the specific architectures remain clean and readable.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare_embeddings(self, states, actions, returns_to_go, costs_to_go, time_steps):
        # create timestep embedding
        timestep_emb = self.timestep_emb(time_steps) if self.time_emb else 0.0 

        raw_state_emb = self.state_emb(states)
        raw_act_emb = self.action_emb(actions)
        
        # concatenate state and action embeddings with timestep embedding
        state_emb = raw_state_emb + timestep_emb 
        act_emb = raw_act_emb + timestep_emb 
        seq_list = [state_emb, act_emb]

        # transform C2G for numerical stability
        if self.cost_transform is not None:
            costs_to_go = self.cost_transform(costs_to_go.detach())

        raw_costs_emb, raw_returns_emb = None, None
        costs_emb, returns_emb = None, None
        
        if self.use_cost:
            raw_costs_emb = self.cost_emb(costs_to_go.unsqueeze(-1))
            costs_emb = raw_costs_emb + timestep_emb
            seq_list.insert(0, costs_emb)
            
        if self.use_rew:
            raw_returns_emb = self.return_emb(returns_to_go.unsqueeze(-1))
            returns_emb = raw_returns_emb + timestep_emb
            seq_list.insert(0, returns_emb)

        raw_embeddings = (raw_state_emb, raw_act_emb, raw_returns_emb, raw_costs_emb)
        time_embeddings = (state_emb, act_emb, returns_emb, costs_emb)
        
        return seq_list, raw_embeddings, time_embeddings

    def _process_transformer(self, seq_list, padding_mask, episode_cost, batch_size, seq_len):
        sequence = torch.stack(seq_list, dim=1).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch_size, self.seq_repeat * seq_len, self.embedding_dim)

        if padding_mask is not None:
            padding_mask = torch.stack([padding_mask] * self.seq_repeat, dim=1).permute(0, 2, 1).reshape(batch_size, -1)

        if self.cost_prefix:
            episode_cost_expanded = episode_cost.unsqueeze(-1).unsqueeze(-1).to(sequence.dtype)
            episode_cost_emb = self.prefix_emb(episode_cost_expanded)
            sequence = torch.cat([episode_cost_emb, sequence], dim=1)
            if padding_mask is not None:
                padding_mask = torch.cat([padding_mask[:, :1], padding_mask], dim=1)

        out = self.emb_drop(self.emb_norm(sequence))
        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        if self.cost_prefix: out = out[:, 1:]

        out = out.reshape(batch_size, seq_len, self.seq_repeat, self.embedding_dim)
        return out.permute(0, 2, 1, 3)

    def _generate_predictions(self, transformer_out, costs_emb):
        action_feature = transformer_out[:, self.seq_repeat - 1]
        state_feat = transformer_out[:, self.seq_repeat - 2]

        if self.use_cost:
            if self.add_cost_feat: state_feat = state_feat + costs_emb.detach()
            if self.mul_cost_feat: state_feat = state_feat * costs_emb.detach()
            if self.cat_cost_feat: state_feat = torch.cat([state_feat, costs_emb.detach()], dim=2)

        action_preds = self.action_head(state_feat)
        cost_preds = F.log_softmax(self.cost_pred_head(action_feature), dim=-1)
        state_preds = self.state_pred_head(action_feature)
        
        return action_preds, cost_preds, state_preds
    
class ContrastiveCDTFront(BaseContrastiveCDT):
    def __init__(self, contrastive_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_dim = contrastive_dim
        
        # MLPs for state and action embeddings
        self.state_emb = nn.Sequential(
            nn.Linear(self.state_dim, self.embedding_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.action_emb = nn.Sequential(
            nn.Linear(self.action_dim, self.embedding_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(inplace=True)
        )
        
        # Project raw embeddings into contrastive dimension
        self.compress = nn.Sequential(
            nn.Linear(4 * self.embedding_dim, self.contrastive_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, states, actions, returns_to_go, costs_to_go, time_steps, padding_mask=None, episode_cost=None, return_latents=False):
        batch_size, seq_len = states.shape[0], states.shape[1]

        seq_list, raw_embs, time_embs = self._prepare_embeddings(states, actions, returns_to_go, costs_to_go, time_steps)
        
        latents = None
        if return_latents:
            # Pass raw embeddings through contrastive component first
            combined_emb = torch.cat(raw_embs, dim=-1)
            latents = self.compress(combined_emb)

        # Get transformer output separately
        transformer_out = self._process_transformer(seq_list, padding_mask, episode_cost, batch_size, seq_len) 
        action_preds, cost_preds, state_preds = self._generate_predictions(transformer_out, time_embs[3])

        if return_latents:
            return action_preds, cost_preds, state_preds, latents
        return action_preds, cost_preds, state_preds
    

class ContrastiveCDTBack(BaseContrastiveCDT):
    def __init__(self, contrastive_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_dim = contrastive_dim
        
        # Project transformer embeddings into contrastive dimension
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.contrastive_dim)
        )

    def forward(self, states, actions, returns_to_go, costs_to_go, time_steps, padding_mask=None, episode_cost=None, return_latents=False):
        batch_size, seq_len = states.shape[0], states.shape[1]

        # prepare embeddings
        seq_list, raw_embs, time_embs = self._prepare_embeddings(states, actions, returns_to_go, costs_to_go, time_steps)
        
        # we only need the time-embedded cost to condition the latents later
        time_c = time_embs[3] 
        
        # pass everything through attention layers FIRST
        transformer_out = self._process_transformer(seq_list, padding_mask, episode_cost, batch_size, seq_len)

        # generate predictions
        action_preds, cost_preds, state_preds = self._generate_predictions(transformer_out, time_c)

        # extract state and action embeddings from transformer layers 
        latents = None
        if return_latents and self.use_cost:
            # transformer_out shape: [Batch, Seq_Len, Modalities, Emb_Dim]
            contextualized_state = transformer_out[:, self.seq_repeat - 2]
            contextualized_action = transformer_out[:, self.seq_repeat - 1]
            
            # cross-condition the transformer outputs with the cost token
            state_conditioned = contextualized_state * time_c
            action_conditioned = contextualized_action * time_c
            
            # final pass back to contrastive head
            latents = self.contrastive_head(state_conditioned + action_conditioned)

        if return_latents:
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
        
        # forward pass through model
        action_preds, cost_preds, state_preds, latents = self.model(
            states=states, actions=actions, returns_to_go=returns,
            costs_to_go=costs_return, time_steps=time_steps, 
            padding_mask=padding_mask, episode_cost=episode_cost, 
            return_latents=True
        )

        # standard CDT losses
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

        # contrastive loss
        if self.cost_boundaries is not None:
            flat_latents = latents[mask > 0]
        
            ep_cost_1d = episode_cost.view(-1)
            traj_labels = torch.bucketize(ep_cost_1d, self.cost_boundaries).long()

            seq_len = states.shape[1]
            batch_labels = traj_labels.unsqueeze(1).expand(-1, seq_len)
            flat_labels = batch_labels[mask > 0].long()
            
            # sub-sample to avoid out-of-memory errors
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
            flat_labels = torch.tensor([0]) 
            
        # combine CDT and contrastive losses
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
        
        if getattr(self, 'stochastic', False):
            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (self.model.temperature() * (entropy - self.model.target_entropy).detach())
            temperature_loss.backward()
            self.log_temperature_optimizer.step()
            
        self.scheduler.step()

        # logging
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