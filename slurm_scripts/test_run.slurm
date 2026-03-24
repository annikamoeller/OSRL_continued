#!/bin/sh
#SBATCH --job-name=cdt_eval_suite
#SBATCH --output=logs/slurm_%j.out
#SBATCH --partition=tue.gpu2.q
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/20234949/thesis/OSRL_continued

eval "$(conda shell.bash hook)"
conda activate CDT_env

export PYTHONPATH=$PYTHONPATH:/home/20234949/thesis/OSRL_continued

python examples/train/train_ccdt.py \
    --task "OfflineAntRun-v0" \
    --project "Codebase_Refactor_Tests" \
    --name "Vanilla_DT_SmokeTest" \
    --contrastive_weight 0.0 \
    --num_buckets 1 \
    --pretrain_steps 0 \
    --device "cuda:0" \
    --update_steps 1000 \
    --eval_every 200 \
    --probe_every 200
