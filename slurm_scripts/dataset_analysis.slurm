#!/bin/bash
#SBATCH --job-name=data_analysis
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/home/20234949/thesis/OSRL_continued
#SBATCH --partition=tue.default.q  # Or whichever CPU partition you prefer

# 1. Environment Setup
eval "$(conda shell.bash hook)"
conda activate CDT_env
export PYTHONPATH=$PYTHONPATH:/home/20234949/thesis/OSRL_continued

echo "🚀 Starting Dataset Analysis Pipeline..."
echo "Running on node: $HOSTNAME"

# 2. Execute the Script
python dataset_analysis/analyze_dataset.py

echo "🎉 Analysis complete! Check the dataset_analysis folder for your outputs."