import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, r2_score
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist

def compute_alignment_uniformity(X, t=2.0):
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    pdist_sq = pdist(X_norm, metric='sqeuclidean')
    return np.mean(pdist_sq), np.log(np.mean(np.exp(-t * pdist_sq)))

@torch.no_grad()
def evaluate_representations(trainer, dataloader, device, step, num_buckets, method_type, save_dir):
    trainer.model.eval()
    all_latents = []
    all_ep_costs = []
    
    # Identify architecture dynamically for clean logging
    model_classname = type(trainer.model).__name__
    arch_label = "Front-Encoder" if "Front" in model_classname else "Back-Encoder"
    
    # Collect a large batch of latents
    for batch in dataloader:
        states, actions, returns, costs_return, time_steps, mask, ep_cost, costs = [b.to(device) for b in batch]
        
        padding_mask = ~mask.to(torch.bool)
        
        # Pass ALL arguments to the model
        _, _, _, latents = trainer.model(
            states=states, 
            actions=actions, 
            returns_to_go=returns, 
            costs_to_go=costs_return, 
            time_steps=time_steps, 
            padding_mask=padding_mask,  
            episode_cost=ep_cost,       
            return_latents=True
        )
        
        valid_latents = latents[mask > 0]
        expanded_ep_cost = ep_cost.unsqueeze(1).expand(-1, latents.shape[1])
        valid_ep_costs = expanded_ep_cost[mask > 0]
    
        all_latents.append(valid_latents.cpu().numpy())
        all_ep_costs.append(valid_ep_costs.cpu().numpy())
        
        if len(all_latents) > 10: 
            break
            
    X = np.concatenate(all_latents, axis=0)
    ep_costs = np.concatenate(all_ep_costs, axis=0)

    is_classification = (method_type == 'bucket')
            
    log_dict = {}

    if is_classification:
        # CATEGORICAL (Bucket Approach)
        boundaries = trainer.cost_boundaries.cpu().numpy()
        y_labels = np.digitize(ep_costs, boundaries)
        
        clf = LogisticRegression(max_iter=1000).fit(X, y_labels)
        log_dict["eval/linear_probe_acc"] = clf.score(X, y_labels)
        log_dict["eval/silhouette_score"] = silhouette_score(X, y_labels) if len(np.unique(y_labels)) > 1 else 0.0
        
        plot_color_data = y_labels
        plot_cmap = 'coolwarm' if num_buckets == 2 else 'viridis'
        cbar_title = f"Safety Severity (0 to {num_buckets - 1})" if num_buckets > 2 else "Safe (0) vs Unsafe (1)"
        
    else:
        # CONTINUOUS (Distance/Threshold Approach)
        reg = LinearRegression().fit(X, ep_costs)
        log_dict["eval/linear_probe_r2"] = r2_score(ep_costs, reg.predict(X))
        
        # Subsample for distance metrics to avoid memory bottlenecks
        idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
        dist_l, dist_c = pdist(X[idx]), pdist(ep_costs[idx].reshape(-1, 1))
        log_dict["eval/spearman_corr"], _ = spearmanr(dist_l, dist_c)
        
        align, uniform = compute_alignment_uniformity(X[idx])
        log_dict["eval/alignment"] = align
        log_dict["eval/uniformity"] = uniform
        
        plot_color_data = ep_costs
        plot_cmap = 'viridis'
        cbar_title = "Trajectory Cost"

    # --- VISUALIZATION ---
    slice_idx = min(2000, len(X))
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X[:slice_idx]) 
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=plot_color_data[:slice_idx], cmap=plot_cmap, alpha=0.6)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label(cbar_title)
    plt.title(f"{arch_label} Latent Space | Step {step}")
    
    # --- DUAL SAVE STRATEGY ---
    if wandb.run is not None:
        log_dict["eval/latent_space"] = wandb.Image(plt)
        wandb.log(log_dict, step=step)
        
        wandb_save_path = os.path.join(wandb.run.dir, f"latents_step_{step}.npz")
        np.savez(wandb_save_path, latents=X, raw_costs=ep_costs, tsne_x=X_tsne[:, 0], tsne_y=X_tsne[:, 1])

    if save_dir is not None:
        local_save_path = os.path.join(save_dir, f"latents_step_{step}.npz")
        np.savez(local_save_path, latents=X, raw_costs=ep_costs, tsne_x=X_tsne[:, 0], tsne_y=X_tsne[:, 1])
    
    plt.close()
    trainer.model.train()