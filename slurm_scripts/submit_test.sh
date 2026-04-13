#!/bin/bash

# 1. Define 2 environments for the quick test
TASKS=( 
    "OfflineCarCircle-v0" 
    "OfflineDroneRun-v0" 
)

# 2. Define the exact configurations to test 
# Format: "EncoderType | ContrastiveWeight | NumBuckets | PretrainSteps | ConfigName"
CONFIGS=(
    "front 0.0 1 0 Front_Baseline"     # Tests Front MLPs without contrastive gradients
    "front 0.1 3 1000 Front_Full"      # Tests Front MLPs WITH contrastive gradients & pretraining
    "back 0.0 1 0 Back_Baseline"       # Tests Back Encoder without contrastive gradients
    "back 0.1 3 1000 Back_Full"        # Tests Back Encoder WITH contrastive gradients & pretraining
)

# Use a single seed for testing
SEED=42

# 3. Define the submission template function
submit_job() {
    local ENV=$1
    local ENCODER=$2
    local WEIGHT=$3
    local BUCKETS=$4
    local PRETRAIN=$5
    local C_NAME=$6
    
    # Determine Batch Size: Drone tasks use 4096; others use 2048.
    local BATCH_SIZE=2048
    if [[ "$ENV" == *"Drone"* ]]; then
        BATCH_SIZE=4096
    fi
    
    local JOB_NAME="Test_${C_NAME}_${ENV}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/stdout/${JOB_NAME}_%j.log
#SBATCH --error=logs/stdout/${JOB_NAME}_%j.err
#SBATCH --time=02:00:00  # Reduced time for a quick 5000 step test
#SBATCH --partition=tue.gpu2.q
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --chdir=/home/20234949/thesis/OSRL_continued

# Load HPC environment
eval "\$(conda shell.bash hook)"
conda activate CDT_env 

# Set paths and environment variables
export DSRL_DATASET_DIR="/home/20234949/thesis/datasets"
export PYTHONPATH="/home/20234949/thesis/OSRL_continued:\$PYTHONPATH"
export MUJOCO_GL="egl"

# Run the training script with quick-check parameters
python examples/train/train_ccdt.py \\
    --task "$ENV" \\
    --seed $SEED \\
    --batch_size $BATCH_SIZE \\
    --device "cuda:0" \\
    --project "OSRL-architecture-tests" \\
    --encoder_type "$ENCODER" \\
    --contrastive_weight $WEIGHT \\
    --pretrain_steps $PRETRAIN \\
    --num_buckets $BUCKETS \\
    --update_steps 5000 \\
    --eval_every 2000 \\
    --probe_every 2000 \\

EOF
    
    echo "Queued: $JOB_NAME (Encoder: $ENCODER, Buckets: $BUCKETS, Pretrain: $PRETRAIN)"
}

# 4. Loop through environments and configurations
echo "================================================"
echo "🚀 Initiating Architecture Stress Test..."
echo "================================================"

for env in "${TASKS[@]}"; do
    echo "------------------------------------------------"
    echo "Submitting jobs for $env"
    echo "------------------------------------------------"

    for config in "${CONFIGS[@]}"; do
        # Parse the configuration string into variables
        read -r ENCODER WEIGHT BUCKETS PRETRAIN C_NAME <<< "$config"
        submit_job "$env" "$ENCODER" "$WEIGHT" "$BUCKETS" "$PRETRAIN" "$C_NAME"
    done
done

echo ""
echo "✅ All 8 test jobs submitted successfully!"
echo "Check wandb project 'OSRL-architecture-tests' to monitor the linear probes."