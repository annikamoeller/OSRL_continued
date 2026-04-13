from dataclasses import dataclass
from examples.configs.cdt_configs import CDTTrainConfig, CDT_DEFAULT_CONFIG

@dataclass
class ContrastiveCDTTrainConfig(CDTTrainConfig):
    # --- THESIS ADDITIONS ---
    num_buckets: int = 2               
    pretrain_steps: int = 0            
    contrastive_dim: int = 64
    contrastive_weight: float = 0.1
    temperature: float = 0.1
    probe_every: int = 5000       
    eval_every: int = 5000     
    # We override update_steps to be 100k by default instead of whatever CDT uses
    update_steps: int = 100_000          
    encoder_type: str = "back"

# We will just map your model to use the exact same tuned defaults as the baseline!
CCDT_DEFAULT_CONFIG = CDT_DEFAULT_CONFIG