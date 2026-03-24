import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

@torch.no_grad()
def evaluate_representations(trainer, dataloader, device, step, num_buckets=2):
    trainer.model.eval()
    all_latents = []
    all_ep_costs = []
    
    # Collect a large batch of latents
    for i, batch in enumerate(dataloader):
        states, actions, returns, costs_return, time_steps, mask, ep_cost, _ = [b.to(device) for b in batch]
        # Get latents from the model
        _, _, _, latents = trainer.model(states, actions, returns, costs_return, time_steps, return_latents=True)
        
        # Mask out padding
        valid_latents = latents[mask > 0]
        
        # Expand episode cost to every valid timestep
        expanded_ep_cost = ep_cost.unsqueeze(1).expand(-1, latents.shape[1])
        valid_ep_costs = expanded_ep_cost[mask > 0]
    
        all_latents.append(valid_latents.cpu().numpy())
        all_ep_costs.append(valid_ep_costs.cpu().numpy())
        
        if len(all_latents) > 10: 
            break
            
    X = np.concatenate(all_latents, axis=0)
    ep_costs = np.concatenate(all_ep_costs, axis=0)

    # Calculate discrete labels only for the metrics, not the plot
    boundaries = trainer.cost_boundaries.cpu().numpy() if hasattr(trainer, 'cost_boundaries') else []
    y_labels = np.digitize(ep_costs, boundaries)
    unique_classes = np.unique(y_labels)

    log_dict = {}

    # 1. & 2. Linear Probing & Silhouette (Only run if we have multiple buckets/classes)
    if len(unique_classes) > 1:
        clf = LogisticRegression(max_iter=1000).fit(X, y_labels)
        log_dict["eval/linear_probe_acc"] = clf.score(X, y_labels)
        log_dict["eval/silhouette_score"] = silhouette_score(X, y_labels)
    else:
        # Fallback for Vanilla DT / Single Bucket
        log_dict["eval/linear_probe_acc"] = 0.0
        log_dict["eval/silhouette_score"] = 0.0
    
    # 3. t-SNE Visualization (Continuous Cost Coloring)
    # We use ep_costs (raw values) instead of y_labels (0, 1, 2)
    tsne = TSNE(n_components=2, random_state=42)
    n_plot = min(len(X), 2000)
    X_tsne = tsne.fit_transform(X[:n_plot]) 
    
    plt.figure(figsize=(9, 7))
    # 'viridis' or 'plasma' are great for continuous cost data
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=ep_costs[:n_plot], cmap='viridis', alpha=0.6)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label("Raw Episode Cost")
    plt.title(f"t-SNE Latent Space (Step {step}) - Colored by Raw Cost")
    
    log_dict["eval/latent_space"] = wandb.Image(plt)
    
    wandb.log(log_dict, step=step)
    
    plt.close()
    trainer.model.train()