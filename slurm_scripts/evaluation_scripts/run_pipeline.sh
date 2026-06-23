#!/usr/bin/env bash
# --- RUN THIS DIRECTLY FROM THE LOGIN NODE ---

REPO_ROOT="/home/20234949/thesis/OSRL_continued"
ABS_EVAL_DIR="${REPO_ROOT}/examples/eval/eval_suite"
PARTITIONS="tue.gpu1.q,tue.gpu2.q,tue.gpu3.q,mcs.gpu.q"

# 1. Parse Command Line Arguments
MODEL_TYPE=${1:-ccdt}   # Options: 'ccdt' or 'vanilla'
EVAL_MODE=${2:-cost}    # Options: 'cost' or 'pareto'

# 2. Dynamic Routing Logic
if [ "$MODEL_TYPE" == "vanilla" ]; then
    LOG_ROOT="${REPO_ROOT}/output_cdt"
    LOG_FILTER="Vanilla_CDT*"
elif [ "$MODEL_TYPE" == "ccdt" ]; then
    LOG_ROOT="${REPO_ROOT}/thesis_final_models"
    LOG_FILTER="*_cw*"
else
    echo "❌ Error: Invalid model type '$MODEL_TYPE'. Use 'ccdt' or 'vanilla'."
    exit 1
fi

if [[ "$EVAL_MODE" != "cost" && "$EVAL_MODE" != "pareto" ]]; then
    echo "❌ Error: Invalid eval mode '$EVAL_MODE'. Use 'cost' or 'pareto'."
    exit 1
fi

echo "🚀 Booting Master Pipeline | Model: ${MODEL_TYPE^^} | Mode: ${EVAL_MODE^^}"

# 3. Initialize clean storage directory
timestamp=$(date +"%Y%m%d_%H%M")
RUN_DIR="${ABS_EVAL_DIR}/${MODEL_TYPE}_${EVAL_MODE}_${timestamp}"
mkdir -p "$RUN_DIR"

# 4. Count matching models
MODEL_COUNT=$(find "$LOG_ROOT" -wholename "*/${LOG_FILTER}/*/config.yaml" | wc -l)

if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "❌ Error: Found 0 matching models under filter ${LOG_FILTER} inside ${LOG_ROOT}"
    exit 1
fi

MAX_IDX=$((MODEL_COUNT - 1))
echo "📊 Found ${MODEL_COUNT} matching models. Provisioning Slurm Array: 0-${MAX_IDX}"

# 5. Launch the Job Array
ARRAY_OUT=$(sbatch -p "$PARTITIONS" --array=0-${MAX_IDX}%30 \
    ${REPO_ROOT}/slurm_scripts/evaluation_scripts/submit_array_eval.sh \
    "$LOG_FILTER" "$RUN_DIR" "$MODEL_TYPE" "$EVAL_MODE")
    
MASTER_JOB_ID=$(echo "$ARRAY_OUT" | awk '{print $NF}')
echo "🧬 Slurm Master Array Job dispatched with ID: $MASTER_JOB_ID"

# 6. Post-Processing Aggregator
mkdir -p "${REPO_ROOT}/slurm_scripts/evaluation_scripts/logs"

sbatch -p "$PARTITIONS" --dependency=afterany:${MASTER_JOB_ID} --job-name=rl_post_process \
       --output=${REPO_ROOT}/slurm_scripts/evaluation_scripts/logs/post_process_%j.out \
       --time=00:15:00 --cpus-per-task=1 --mem=4G --wrap="
          # Initialize environment inside the new compute node
          eval \"\$(conda shell.bash hook)\"
          conda activate CDT_env
          export PYTHONPATH=\$PYTHONPATH:${REPO_ROOT}
          
          cd ${REPO_ROOT}
          
          # Consolidated Pandas merger
          python -c \"
import os, glob, shutil, pandas as pd

run_dir = '${RUN_DIR}'
parts_dir = os.path.join(run_dir, 'parts')

part_files = sorted(glob.glob(os.path.join(run_dir, 'part_*.csv')))

if part_files:
    df = pd.concat([pd.read_csv(f) for f in part_files], ignore_index=True)
    df.to_csv(os.path.join(run_dir, 'raw_data.csv'), index=False)
    print(f'🎉 Consolidated {len(part_files)} worker chunks into raw_data.csv')
    
    os.makedirs(parts_dir, exist_ok=True)
    for f in part_files:
        shutil.move(f, os.path.join(parts_dir, os.path.basename(f)))
    print(f'📂 Sequestered source chunks into: {parts_dir}/')
else:
    print('❌ No worker part files found to consolidate.')
          \"
          
          # Automatically trigger plotting using the unified eval_suite CLI
          echo '📈 Generating visual charts...'
          python ${ABS_EVAL_DIR}/eval_suite.py plot_${EVAL_MODE} --run_dir ${RUN_DIR}
       "

echo "🎉 Pipeline aggregation node queued up successfully!"