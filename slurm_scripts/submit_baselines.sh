#!/bin/bash

# ==========================================
# BASELINE CONFIGURATION: VANILLA CDT
# ==========================================
WANDB_PROJECT="Vanilla_CDT_Baselines"

# Define the environments you want to test
ENVS=(
    "OfflineAntRun-v0" 
    "OfflineCarCircle-v0" 
    "OfflineCarRun-v0" 
    "OfflineDroneCircle-v0" 
    "OfflineDroneRun-v0" 
)

# 3 Seeds for a rigorous control group
SEEDS=(8 42 123)

mkdir -p slurm_logs

# ==========================================
# SUBMISSION LOOP
# ==========================================
for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do 
        
        # Ensure log directory exists for vanilla CDT console output
        LOG_DIR="logs/stdout_vanilla_cdt"
        mkdir -p "$LOG_DIR"

        # Set up names to keep things separate from CCDT runs
        JOB_NAME="cdt_vanilla_${ENV}_s${SEED}"
        GROUP_NAME="Vanilla_CDT_${ENV}"
        
        echo "Submitting Baseline: $JOB_NAME"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --partition=tue.gpu1.q,tue.gpu2.q,tue.gpu3.q,mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/20234949/thesis/OSRL_continued

# Load environment
source ~/.bashrc
conda activate CDT_env
export DSRL_DATASET_DIR="/home/20234949/thesis/datasets"
export PYTHONPATH="/home/20234949/thesis/OSRL_continued:\$PYTHONPATH"

# Execute Vanilla CDT Training (Calls the default OSRL training script)
python examples/train/train_cdt.py \
    --task $ENV \
    --seed $SEED \
    --project $WANDB_PROJECT \
    --group "$GROUP_NAME" \
    --eval_every 5000 \
    --device "cuda:0" \
    --logdir "output_cdt"
EOF

    done
done

echo "🎉 All Vanilla CDT baseline jobs (3 seeds each) submitted!"