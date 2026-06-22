#!/usr/bin/env bash

# Define your sweep parameters
ENVS=("OfflineDroneRun-v0")
SEEDS=(8 42)
METHODS=("threshold" "distance") 
CONTRASTIVE_WEIGHT=0.5
STEPS=75000

for METHOD in "${METHODS[@]}"; do
    for ENV in "${ENVS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            
            JOB_NAME="ccdt_${METHOD}_${ENV}_s${SEED}"
            echo "🚀 Submitting: Env=$ENV | Seed=$SEED | Method=$METHOD | Weight=$CONTRASTIVE_WEIGHT"

            # The 'sbatch <<EOT' block STARTS HERE. Everything below this goes to SLURM.
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

python -u examples/train/train_ccdt.py \
    --task ${ENV} \
    --encoder_type back \
    --contrastive_type ${METHOD} \
    --cost_threshold 10.0 \
    --contrastive_weight ${CONTRASTIVE_WEIGHT} \
    --seed ${SEED} \
    --update_steps ${STEPS} \
    --eval_every 5000 \
    --probe_every 5000 \
    --project CCDT_Test_Runs \
    --group Sweep_${METHOD}_cw${CONTRASTIVE_WEIGHT} \
    --device "cuda:0" \
    --logdir "arch_b_c"
EOT
            
            sleep 1
            
        done
    done
done