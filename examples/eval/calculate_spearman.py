import os
import json
import torch
import numpy as np
from scipy.stats import spearmanr

def compute_spearman_for_all_tables(run_path, device='cuda'):
    print(f"DEBUG: Searching for all .table.json files in {run_path}")
    
    # Collect all found files
    all_table_files = []
    for root, dirs, files in os.walk(run_path):
        for file in files:
            if file.endswith('.table.json'):
                all_table_files.append(os.path.join(root, file))
    
    if not all_table_files:
        print(f"DEBUG: No .table.json files found.")
        return
    
    print(f"DEBUG: Found {len(all_table_files)} tables. Processing...")
    
    for table_path in all_table_files:
        print(f"\n--- Processing Table: {os.path.basename(table_path)} ---")
        
        with open(table_path, 'r') as f:
            content = json.load(f)
        
        if 'data' not in content:
            continue
            
        raw_data = np.array(content['data'])
        latents = raw_data[:, 0:2]  # columns: tsne_x, tsne_y
        costs = raw_data[:, 2]      # column: raw_cost
        
        # GPU Acceleration Setup
        sample_size = min(2000, len(latents))
        idx = np.random.choice(len(latents), sample_size, replace=False)
        
        X = torch.tensor(latents[idx], device=device, dtype=torch.float32)
        C = torch.tensor(costs[idx], device=device, dtype=torch.float32).unsqueeze(1)
        
        dist_latent = torch.cdist(X, X, p=2)
        dist_cost = torch.cdist(C, C, p=1)
        
        triu_indices = torch.triu_indices(sample_size, sample_size, offset=1)
        latent_flat = dist_latent[triu_indices[0], triu_indices[1]]
        cost_flat = dist_cost[triu_indices[0], triu_indices[1]]
        
        corr, _ = spearmanr(latent_flat.cpu().numpy(), cost_flat.cpu().numpy())
        print(f"Spearman Correlation: {corr:.4f}")

# Paths
runs = {
    "Distance": "/vast.mnt/home/20234949/thesis/OSRL_continued/wandb/run-20260621_120002-fa39050c-5d4f-48b7-b056-881fe8cc1180",
    "Threshold": "/vast.mnt/home/20234949/thesis/OSRL_continued/wandb/run-20260621_115813-91cbc88d-bbfb-4519-8bd4-e567363ddf26"
}

for name, path in runs.items():
    print(f"\n===== SCANNING RUN: {name} =====")
    compute_spearman_for_all_tables(path)