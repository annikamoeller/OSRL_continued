#!/bin/bash

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
ARCH="back"
WANDB_PROJECT="CCDT_Back_Architecture"

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

# ==========================================
# SUBMISSION LOOP
# ==========================================
for ENV in "${ENVS[@]}"; do
    for B in "${BUCKETS[@]}"; do
        
        JOB_NAME="back_${ENV}_${B}B"
        
        echo "Submitting: $JOB_NAME"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=logs/stdout/%x_%j.out
#SBATCH --error=logs/stdout/%x_%j.err
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
    --group "Bucket_Sweep_${ENV}" \
    --eval_every 5000 \
    --probe_every 5000 \
    --device "cuda:0"
EOF

    done
done

echo "🎉 All Back-Encoder jobs submitted to the dynamic queue!"