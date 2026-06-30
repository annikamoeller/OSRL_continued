#!/bin/bash

# --- SWEEP CONFIGURATION ---
ARCHITECTURES=("vanilla" "ccdt_bucket")
SEEDS=(8)
ENV_NAMES=("OfflineCarCircle-v0")
SKEW_TYPES=("Extreme_Imbalance")
# "SubOptimal_Safe" "Reckless_Expert")

# Ensure the logs directory exists
mkdir -p logs

echo "🔥 Launching Skewed Dataset Training Sweeps..."

for ENV_NAME in "${ENV_NAMES[@]}"; do
    
    # Match the environment to its specific HDF5 filename
    if [[ "$ENV_NAME" == "OfflineAntRun-v0" ]]; then
        DATASET_FILE="SafetyAntRun-v0-150-1816.hdf5"
    elif [[ "$ENV_NAME" == "OfflineCarCircle-v0" ]]; then
        DATASET_FILE="SafetyCarCircle-v0-100-1450.hdf5"
    elif [[ "$ENV_NAME" == "OfflineDroneRun-v0" ]]; then
        DATASET_FILE="SafetyDroneRun-v0-140-1990.hdf5"
    else
        echo "⚠️ Unknown environment: $ENV_NAME. Skipping."
        continue
    fi

    for SKEW in "${SKEW_TYPES[@]}"; do
        # Reconstruct the absolute path to the newly generated splits
        SPLIT="/home/20234949/thesis/datasets/modified_datasets/${SKEW}/${DATASET_FILE}"
        
        for ARCH in "${ARCHITECTURES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                
                echo "Submitting: $ARCH | Env: $ENV_NAME | Split: $SKEW | Seed: $SEED"
                
                # Submit the job and pass the arguments to the slurm template
                sbatch train_skew.slurm "$ARCH" "$SPLIT" "$SEED" "$ENV_NAME" "$SKEW"
                
                # Brief pause to avoid hammering the Slurm scheduler
                sleep 1
                
            done
        done
    done
done

echo "✅ All jobs submitted to the cluster!"