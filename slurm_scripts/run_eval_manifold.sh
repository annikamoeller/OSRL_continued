#!/usr/bin/env bash
#SBATCH --job-name=eval_manifold
#SBATCH --partition=tue.gpu1.q,tue.gpu2.q,mcs.gpu.q
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=/home/20234949/thesis/OSRL_continued/slurm_scripts/logs/eval_manifold_%j.out
#SBATCH --error=/home/20234949/thesis/OSRL_continued/slurm_scripts/logs/eval_manifold_%j.err
#SBATCH --chdir=/home/20234949/thesis/OSRL_continued

# Fail fast if anything crashes
set -e

echo "🚀 Booting Manifold Evaluation..."
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

# 1. Load Conda environment
eval "$(conda shell.bash hook)"
conda activate CDT_env

# 2. Set Paths
export DSRL_DATASET_DIR="/home/20234949/thesis/datasets"
export PYTHONPATH="/home/20234949/thesis/OSRL_continued:$PYTHONPATH"

# 3. Execute the Evaluation Script
# Running with -u for real-time logging. 
# Adjust the .py filename below if you saved it under a different name!
python -u examples/eval/eval_suite/plot_hairball.py \
    --models both \
    --envs AntRun CarCircle DroneRun \
    --trajectories 200

echo "🏁 Evaluation complete. Plots should be in examples/eval/eval_suite/plots"


