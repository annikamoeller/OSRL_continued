#!/usr/bin/env bash

# Define target environments and evaluation seeds
ENVS=("OfflineAntRun-v0" "OfflineCarCircle-v0" "OfflineDroneRun-v0")
SEEDS=(0)
CONTRASTIVE_WEIGHT=0.0
STEPS=75000

# Ensure log directory exists
mkdir -p /home/20234949/thesis/OSRL_continued/slurm_scripts/logs

for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        # Tagging the job explicitly
        JOB_NAME="ablation_ccdt_${ENV}_s${SEED}"
        echo "🚀 Submitting Strict Ablation: Env=$ENV | Seed=$SEED | Weight=$CONTRASTIVE_WEIGHT"

        sbatch <<EOT
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=tue.gpu1.q,tue.gpu2.q,mcs.gpu.q
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=/home/20234949/thesis/OSRL_continued/slurm_scripts/logs/${JOB_NAME}_%j.out
#SBATCH --error=/home/20234949/thesis/OSRL_continued/slurm_scripts/logs/${JOB_NAME}_%j.err
#SBATCH --chdir=/home/20234949/thesis/OSRL_continued

set -e

eval "\$(conda shell.bash hook)"
conda activate CDT_env
export DSRL_DATASET_DIR="/home/20234949/thesis/datasets"
export PYTHONPATH="/home/20234949/thesis/OSRL_continued:\$PYTHONPATH"

# Forcing architectural equivalence to the Vanilla Baseline
python -u examples/train/train_ccdt.py \
    --task ${ENV} \
    --encoder_type back \
    --contrastive_type threshold \
    --cost_threshold 10.0 \
    --contrastive_weight ${CONTRASTIVE_WEIGHT} \
    --seed ${SEED} \
    --batch_size 2048 \
    --seq_len 10 \
    --num_heads 8 \
    --update_steps ${STEPS} \
    --eval_every 5000 \
    --probe_every 5000 \
    --project CCDT_Thesis_Ablations \
    --group Strict_Ablation_cw0.0 \
    --device "cuda:0" \
    --logdir "thesis_final_models"
EOT
        
        # Slight delay to respect the SLURM scheduler queue
        sleep 1
        
    done
done
