#!/bin/bash
#SBATCH --job-name=skew_train
#SBATCH --output=logs/skew_%j.out
#SBATCH --error=logs/skew_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=tue.gpu1.q,tue.gpu2.q,mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Initialize conda and activate
source ~/.bashrc
conda activate CDT_env

# Grab arguments
ARCH=$1
SPLIT_PATH=$2
SEED=$3
ENV_NAME=$4
SKEW=$5 # 

# Move to project root
cd /home/20234949/thesis/OSRL_continued/
export PYTHONPATH=$(pwd)

# IMPORTANT: Force CUDA to see ONLY the single GPU allocated by Slurm.
export CUDA_VISIBLE_DEVICES=0

echo "🚀 Starting Job ID: $SLURM_JOB_ID"
echo "🧠 Architecture: $ARCH"
echo "🌱 Seed: $SEED"
echo "----------------------------------------"

# Pre-flight check: Verify PyTorch can see the GPU
python -c "import torch; print(f'Verification - CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

# Setup training arguments
if [ "$ARCH" == "vanilla" ]; then
    TRAIN_CMD="python examples/train/train_cdt.py \
        --task $ENV_NAME \
        --dataset $SPLIT_PATH \
        --seed $SEED \
        --num_workers 4 \
        --device cuda:0 \
        --name ${ARCH}_${ENV_NAME}_${SKEW}_seed${SEED}"
        
elif [ "$ARCH" == "ccdt_bucket" ]; then
    TRAIN_CMD="python examples/train/train_ccdt.py \
        --task $ENV_NAME \
        --dataset $SPLIT_PATH \
        --seed $SEED \
        --encoder_type back \
        --contrastive_type bucket \
        --num_buckets 3 \
        --contrastive_weight 3.0 \
        --num_workers 4 \
        --device cuda:0 \
        --name ${ARCH}_${ENV_NAME}_${SKEW}_seed${SEED}"
        
else
    echo "❌ Unknown architecture: $ARCH"
    exit 1
fi

# Execute the run
echo "Running: $TRAIN_CMD"
eval $TRAIN_CMD

echo "✅ Job completed successfully."