#!/bin/bash
#SBATCH --chdir=/home/20234949/thesis/OSRL_continued
#SBATCH --job-name=spearman_calc
#SBATCH --output=slurm_scripts/evaluation_scripts/spearman_logs/spearman_calc.out
#SBATCH --error=slurm_scripts/evaluation_scripts/spearman_logs/spearman_calc.err
#SBATCH --partition=tue.gpu1.q,tue.gpu2.q,mcs.gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

eval "$(conda shell.bash hook)"

# Activate your environment
conda activate CDT_env

# Run the python script
python examples/eval/calculate_spearman.py