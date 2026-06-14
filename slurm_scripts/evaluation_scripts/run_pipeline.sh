#!/usr/bin/bash
# --- RUN THIS DIRECTLY FROM THE LOGIN NODE ---

REPO_ROOT="/home/20234949/thesis/OSRL_continued"
# 🌟 FIXED: Pointing to your clean path so find can actually locate the 166 models
LOG_ROOT="${REPO_ROOT}/thesis_final_models"
LOG_FILTER="*_cw*" # 🌟 FIXED: Simplified to match the parent folders perfectly
ABS_EVAL_DIR="${REPO_ROOT}/examples/eval/eval_suite"
PARTITIONS="tue.gpu1.q,tue.gpu2.q,tue.gpu3.q,mcs.gpu.q"

# 1. Initialize our clean results storage directory
timestamp=$(date +"%Y%m%d_%H%M")
RUN_DIR="${ABS_EVAL_DIR}/eval_${timestamp}"
mkdir -p "$RUN_DIR"

# 2. Count matching models
MODEL_COUNT=$(find "$LOG_ROOT" -wholename "*/${LOG_FILTER}/*/config.yaml" | wc -l)

if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "❌ Error: Found 0 matching models under filter ${LOG_FILTER} inside ${LOG_ROOT}"
    exit 1
fi

MAX_IDX=$((MODEL_COUNT - 1))
echo "📊 Found ${MODEL_COUNT} matching models. Provisioning Slurm Array: 0-${MAX_IDX}"

# 3. Launch the Job Array
ARRAY_OUT=$(sbatch -p "$PARTITIONS" --array=0-${MAX_IDX}%30 ${REPO_ROOT}/slurm_scripts/evaluation_scripts/submit_array_eval.sh "$LOG_FILTER" "$RUN_DIR")
MASTER_JOB_ID=$(echo "$ARRAY_OUT" | grep -o '[0-9]*')

echo "🧬 Slurm Master Array Job dispatched with ID: $MASTER_JOB_ID"

# 4. Compile part files, sequester into parts/, and run downstream scripts
mkdir -p "${REPO_ROOT}/slurm_scripts/evaluation_scripts/logs"

sbatch -p "$PARTITIONS" --dependency=afterok:${MASTER_JOB_ID} --job-name=cdt_post_process \
       --output=${REPO_ROOT}/slurm_scripts/evaluation_scripts/logs/post_process_%j.out \
       --time=00:15:00 --cpus-per-task=1 --mem=4G --wrap="
          cd ${REPO_ROOT}
          
          # Consolidated Pandas merger + auto-move isolation loop
          python -c \"
import os, glob, shutil, pandas as pd

run_dir = '${RUN_DIR}'
parts_dir = os.path.join(run_dir, 'parts')

# Gather all raw evaluation slices from workers
part_files = sorted(glob.glob(os.path.join(run_dir, '*.csv')))
part_files = [f for f in part_files if 'raw_data.csv' not in f and os.path.dirname(f) != parts_dir]

if part_files:
    # 1. Combine dataframes cleanly
    df = pd.concat([pd.read_csv(f) for f in part_files], ignore_index=True)
    df.to_csv(os.path.join(run_dir, 'raw_data.csv'), index=False)
    print(f'🎉 Consolidated {len(part_files)} worker chunks into raw_data.csv')
    
    # 2. Sequester individual parts to isolate the directory workspace
    os.makedirs(parts_dir, exist_ok=True)
    for f in part_files:
        shutil.move(f, os.path.join(parts_dir, os.path.basename(f)))
    print(f'📂 Successfully sequestered all source files into: {parts_dir}/')
else:
    print('❌ No worker part files found to consolidate.')
          \"
          
          # Fire downstream visualization engines over clean root path
          python ${ABS_EVAL_DIR}/plot_eval.py ${RUN_DIR}
          python ${ABS_EVAL_DIR}/table_eval.py ${RUN_DIR}
       "

echo "🎉 Post-processing aggregation node queued up safely with auto-cleanup handling!"