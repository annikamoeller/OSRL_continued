from dataclasses import dataclass
from examples.configs.cdt_configs import CDTTrainConfig, CDT_DEFAULT_CONFIG

@dataclass
class ContrastiveCDTTrainConfig(CDTTrainConfig):
    num_buckets: int = 2               
    pretrain_steps: int = 0            
    contrastive_dim: int = 64
    contrastive_weight: float = 0.1
    temperature: float = 0.1
    probe_every: int = 5000       
    eval_every: int = 5000     
    update_steps: int = 100_000          
    encoder_type: str = "back"
    # Add these three lines to your ContrastiveCDTTrainConfig class:
    contrastive_type: str = "bucket" # "bucket", "threshold", or "distance"
    alpha: float = 0.02              # For distance matching
    cost_threshold: float = 10.0     # For threshold infonce

CCDT_DEFAULT_CONFIG = CDT_DEFAULT_CONFIG