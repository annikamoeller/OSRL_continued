#!/bin/bash
#SBATCH --job-name=cdt_eval_vanilla
#SBATCH --output=slurm_scripts/evaluation_scripts/logs/eval_vanilla_%j.out
#SBATCH --partition=tue.gpu1.q,tue.gpu2.q,tue.gpu3.q,mcs.gpu.q
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --chdir=/home/20234949/thesis/OSRL_continued

# --- CONFIGURATION ---
LOG_FILTER="Vanilla_CDT_*" 
TARGET_SCRIPT="examples/eval/eval_suite/collect_vanilla_eval.py"

# Set up cluster runtime environments
eval "$(conda shell.bash hook)"
conda activate CDT_env
export PYTHONPATH=$PYTHONPATH:/home/20234949/thesis/OSRL_continued

echo "🚀 Starting Sequential Evaluation Pipeline for Vanilla Baselines: $LOG_FILTER"

# Ensure our local log output directory exists so Slurm doesn't bounce the job descriptor
mkdir -p slurm_scripts/evaluation_scripts/logs

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "❌ Error: Could not find target script layout at $TARGET_SCRIPT"
    exit 1
fi

# --- SAFE CONFIG OVERRIDE ---
# Anchor to the literal definition line to route tracking paths directly into output_cdt
sed -i 's|^LOG_ROOT = .*|LOG_ROOT = "/home/20234949/thesis/OSRL_continued/output_cdt"|g' "$TARGET_SCRIPT"

# Run your dedicated vanilla evaluation script
python "$TARGET_SCRIPT" --log_filter "$LOG_FILTER" | tee temp_eval_output.log

# Extract the unique timestamped directory string generated during execution
RUN_DIR=$(grep -o 'examples/eval/eval_suite/eval_vanilla_[0-9_]*' temp_eval_output.log | head -n 1)

if [ -z "$RUN_DIR" ]; then
    echo "❌ Error: Could not resolve target run directory from streaming log matrix. Check logs above."
    rm -f temp_eval_output.log
    exit 1