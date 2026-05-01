#!/bin/bash
#SBATCH --job-name=cdt_eval_cw04
#SBATCH --output=logs/eval_cw04_%j.out
#SBATCH --partition=tue.gpu1.q,tue.gpu2.q,tue.gpu3.q,mcs.gpu.q
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --chdir=/home/20234949/thesis/OSRL_continued

# --- CONFIGURATION ---
# Match the folder naming pattern used in your training submit scripts
LOG_FILTER="Bucket_Sweep_cw04_*" 

eval "$(conda shell.bash hook)"
conda activate CDT_env
export PYTHONPATH=$PYTHONPATH:/home/20234949/thesis/OSRL_continued

echo "🚀 Starting Evaluation Pipeline for: $LOG_FILTER"

# Run Data Collection with the custom filter
python examples/eval/eval_suite/collect_eval_v2.py --log_filter "$LOG_FILTER" | tee temp_eval_output.log

# Extract the Run Directory
RUN_DIR=$(grep -o 'examples/eval/eval_suite/eval_[0-9_]*' temp_eval_output.log | tail -n 1)

if [ -z "$RUN_DIR" ]; then
    echo "❌ Error: No experiments found. Check if your LOG_FILTER matches your log folders."
    exit 1
fi

echo "📂 Run Directory Captured: $RUN_DIR"

# Run Plotting and Tables
python examples/eval/eval_suite/plot_eval.py "$RUN_DIR"
python examples/eval/eval_suite/table_eval.py "$RUN_DIR"

rm temp_eval_output.log
echo "🎉 Pipeline Complete!"