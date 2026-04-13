#!/bin/sh
#SBATCH --job-name=cdt_eval_suite
#SBATCH --output=logs/slurm_%j.out
#SBATCH --partition=tue.gpu1.q,tue.gpu2.q,tue.gpu3.q,mcs.gpu.q
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --chdir=/home/20234949/thesis/OSRL_continued

# 1. Environment Setup
eval "$(conda shell.bash hook)"
conda activate CDT_env
export PYTHONPATH=$PYTHONPATH:/home/20234949/thesis/OSRL_continued

echo "🚀 Starting Full Evaluation Pipeline..."

# 2. Run Data Collection
# (Assuming you saved the collection script as collect_eval.py)
# We use 'tee' to print to the Slurm log AND a temporary file
python examples/eval/eval_suite/collect_eval_v2.py | tee temp_eval_output.log

# 3. Extract the Run Directory
# We grep the temporary file to find the exact timestamped folder that was created
RUN_DIR=$(grep -o 'examples/eval/eval_suite/eval_[0-9_]*' temp_eval_output.log | tail -n 1)

# Safety check in case the collection script failed
if [ -z "$RUN_DIR" ]; then
    echo "❌ Error: Could not find the generated run directory. Check the logs above."
    rm temp_eval_output.log
    exit 1
fi

echo ""
echo "📂 Run Directory Captured: $RUN_DIR"

# 4. Run Plotting
echo "📈 Generating Plots..."
python examples/eval/eval_suite/plot_eval.py "$RUN_DIR"

# 5. Run Table Generation
echo "📊 Generating Publication Tables..."
python examples/eval/eval_suite/table_eval.py "$RUN_DIR"

# 6. Cleanup
rm temp_eval_output.log
echo "🎉 Pipeline Complete! Check $RUN_DIR for your CSVs, PNGs, and Tables."