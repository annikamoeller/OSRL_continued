import os
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

    # Calculate discrete labels
    boundaries = trainer.cost_boundaries.cpu().numpy() if hasattr(trainer, 'cost_boundaries') else []
    y_labels = np.digitize(ep_costs, boundaries)
    unique_classes = np.unique(y_labels)

    log_dict = {}

    # 1. & 2. Linear Probing & Silhouette
    if len(unique_classes) > 1:
        clf = LogisticRegression(max_iter=1000).fit(X, y_labels)
        log_dict["eval/linear_probe_acc"] = clf.score(X, y_labels)
        log_dict["eval/silhouette_score"] = silhouette_score(X, y_labels)
    else:
        log_dict["eval/linear_probe_acc"] = 0.0
        log_dict["eval/silhouette_score"] = 0.0
    
    # 3. t-SNE Visualization 
    tsne = TSNE(n_components=2, random_state=42)
    n_plot = min(len(X), 2000)
    X_tsne = tsne.fit_transform(X[:n_plot]) 
    ep_costs_plot = ep_costs[:n_plot]
    y_labels_plot = y_labels[:n_plot]
    
    # --- ADDITION 1: Log as a WandB Table ---
    table_data = [[X_tsne[i, 0], X_tsne[i, 1], ep_costs_plot[i], y_labels_plot[i]] for i in range(n_plot)]
    log_dict["eval/tsne_data_table"] = wandb.Table(
        data=table_data, columns=["tsne_x", "tsne_y", "raw_cost", "bucket_label"]
    )

    # --- ADDITION 2: Save Locally ---
    if wandb.run is not None:
        save_path = os.path.join(wandb.run.dir, f"tsne_arrays_step_{step}.npz")
        np.savez(save_path, tsne_x=X_tsne[:, 0], tsne_y=X_tsne[:, 1], 
                 raw_costs=ep_costs_plot, labels=y_labels_plot)

    # Create an explicit Figure and Axis object
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=ep_costs_plot, cmap='viridis', alpha=0.6)
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Raw Episode Cost")
    ax.set_title(f"t-SNE Latent Space (Step {step}) - Colored by Raw Cost")
    
    # Pass the specific 'fig', NOT the global 'plt'
    log_dict["eval/latent_space"] = wandb.Image(fig)
    
    # (Optional: If you are running Multi-GPU, you can wrap this log in an 'if rank == 0:' check)
    wandb.log(log_dict, step=step)
    
    # Explicitly destroy this exact figure object to free memory
    plt.close(fig)
    trainer.model.train()
    trainer.model.train()