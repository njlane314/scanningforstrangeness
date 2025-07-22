#!/bin/bash

source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env

ROOT_FILE="./filtered_numu_cc_events.root"
PYTHON_SCRIPT="train_classifier.py"
OUTPUT_WEIGHTS="./classifier_model.pth"

TREE_NAMES="mc_inclusive_run1_fhc,mc_strangeness_run1_fhc"

LEARNING_RATE=0.0001
BATCH_SIZE=64
MAX_STEPS=10000
VAL_FREQUENCY=1000
NUM_WORKERS=0
SEED=48

python "${PYTHON_SCRIPT}" \
    --root_file_path "${ROOT_FILE}" \
    --tree_names "${TREE_NAMES}" \
    --plane "u" \
    --lr "${LEARNING_RATE}" \
    --batch_size "${BATCH_SIZE}" \
    --max_steps "${MAX_STEPS}" \
    --val_freq "${VAL_FREQUENCY}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --weights "${OUTPUT_WEIGHTS}"

echo "Training complete. Model weights saved to ${OUTPUT_WEIGHTS}"