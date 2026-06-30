#!/bin/bash

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
ARCHS=("front" "back")

# Define the environments you want to test
ENVS=(
    # "OfflineAntRun-v0" 
    # "OfflineCarCircle-v0" 
    "OfflineDroneRun-v0" 
)

# Define the hyperparameters to sweep
BUCKETS=(2 3 5)
SEEDS=(8 42)
WEIGHTS=(0.1 0.3) 

mkdir -p slurm_logs

# ==========================================
# SUBMISSION LOOP
# ==========================================
for ARCH in "${ARCHS[@]}"; do
    for ENV in "${ENVS[@]}"; do
        for B in "${BUCKETS[@]}"; do
            for SEED in "${SEEDS[@]}"; do 
                for CW in "${WEIGHTS[@]}"; do 
                    
                    # Clean up the weight string (e.g., 0.4 -> 04)
                    CW_CLEAN="${CW//./}"
                    
                    # Capitalize the architecture for the W&B project name (Front/Back)
                    if [ "$ARCH" == "front" ]; then
                        ARCH_CAP="Front"
                    else
                        ARCH_CAP="Back"
                    fi

                    # Ensure log directory exists for this specific architecture and weight
                    LOG_DIR="logs/stdout_${ARCH}_cw${CW_CLEAN}"
                    mkdir -p "$LOG_DIR"

                    # Dynamically set W&B project and Job name
                    WANDB_PROJECT="CCDT_${ARCH_CAP}_Architecture_cw${CW_CLEAN}"
                    JOB_NAME="${ARCH}_cw${CW_CLEAN}_${ENV}_${B}B_s${SEED}"
                    
                    echo "Submitting: $JOB_NAME (Weight: $CW)"

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

# Execute Training (Single line format avoids any hidden space or multi-line string glitches)
python examples/train/train_ccdt.py --task $ENV --seed $SEED --encoder_type $ARCH --num_buckets $B --project $WANDB_PROJECT --group "${ARCH_CAP}_${ENV}_${B}Buckets_cw${CW_CLEAN}" --eval_every 5000 --probe_every 5000 --contrastive_weight $CW --device "cuda:0" --logdir "output"
EOF

                done
            done
        done
    done
done

echo "🎉 All jobs (Arch + Weight + Bucket + Seed sweep) submitted!"