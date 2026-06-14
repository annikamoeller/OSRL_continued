#!/bin/bash
#SBATCH --job-name=cdt_array_eval
#SBATCH --output=/home/20234949/thesis/OSRL_continued/slurm_scripts/evaluation_scripts/logs/array_%A_%a.out
#SBATCH --partition=tue.gpu1.q,tue.gpu2.q,tue.gpu3.q,mcs.gpu.q
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --chdir=/home/20234949/thesis/OSRL_continued

EXP_FILTER=$1
RUN_DIR=$2

# Load your cluster configurations safely
eval "$(conda shell.bash hook)"
conda activate CDT_env
export PYTHONPATH=$PYTHONPATH:/home/20234949/thesis/OSRL_continued

echo "🧬 Array worker active for Task Index: $SLURM_ARRAY_TASK_ID"

# Execute your clean dispatcher wrapper script
python /home/20234949/thesis/OSRL_continued/slurm_scripts/evaluation_scripts/collect_array.py \
    --log_filter "$EXP_FILTER" \
    --array_idx $SLURM_ARRAY_TASK_ID \
    --run_dir "$RUN_DIR"