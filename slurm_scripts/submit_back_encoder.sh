#!/bin/bash

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
ARCH="back"
# 🚨 UPDATED: Appended cw04 to isolate from the 0.1 runs
WANDB_PROJECT="CCDT_Back_Architecture_cw04"

# Define the environments you want to test
ENVS=(
    "OfflineAntRun-v0" 
    "OfflineCarCircle-v0" 
    "OfflineCarRun-v0" 
    "OfflineDroneCircle-v0" 
    "OfflineDroneRun-v0" 
)

# Define the number of buckets to sweep
BUCKETS=(2 3 5 8 10)

mkdir -p slurm_logs
# 🚨 UPDATED: Ensure the new log directory exists
mkdir -p logs/stdout_cw04

# ==========================================
# SUBMISSION LOOP
# ==========================================
for ENV in "${ENVS[@]}"; do
    for B in "${BUCKETS[@]}"; do
        
        # 🚨 UPDATED: Job name now reflects the cw04 run
        JOB_NAME="back_cw04_${ENV}_${B}B"
        
        echo "Submitting: $JOB_NAME"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
# 🚨 UPDATED: Rerouting stdout/error logs to avoid overwriting old ones
#SBATCH --output=logs/stdout_cw04/%x_%j.out
#SBATCH --error=logs/stdout_cw04/%x_%j.err
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

# Execute Training
python examples/train/train_ccdt.py \
    --task $ENV \
    --seed 8 \
    --encoder_type $ARCH \
    --num_buckets $B \
    --project $WANDB_PROJECT \
    --group "Bucket_Sweep_cw04_${ENV}" \
    --eval_every 5000 \
    --probe_every 5000 \
    --contrastive_weight 0.4 \
    --device "cuda:0"
EOF

    done
done

echo "🎉 All Back-Encoder cw0.4 jobs submitted to the dynamic queue!"