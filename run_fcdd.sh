#!/bin/bash

# ==============================================================================
# ## --- Environment Setup --- ##
# ==============================================================================
# Activate your conda environment. Modify the path if necessary.
echo "Activating Conda environment..."
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env


# ==============================================================================
# ## --- Configuration --- ##
# ==============================================================================
# Define the path to your data and where to save/load the model weights.
ROOT_FILE="./filtered_numu_cc_events.root"
OUTPUT_WEIGHTS="./fcdd_model_weights.pth"
PYTHON_SCRIPT="train_fcdd.py" # The name of the new Python script

# --- Define Data Sources for Anomaly Detection ---
# Background events are treated as NOMINAL data (label 0)
BACKGROUND_TREES=(
    "mc_inclusive_run1_fhc"
)

# Signal events are treated as ANOMALOUS data (label 1)
SIGNAL_TREES=(
    "mc_strangeness_run1_fhc"
)


# ==============================================================================
# ## --- EXECUTION BLOCK --- ##
# ==============================================================================
# Choose whether to run in 'train' or 'test' mode.
# Comment out the block you are not using.

# --- 1. TRAINING MODE ---
# This command starts the training process from scratch.
# It uses both background and signal trees to learn the difference.
echo "Starting FCDD training..."
python "${PYTHON_SCRIPT}" \
    --test \
    --root_file_path "${ROOT_FILE}" \
    --weights "${OUTPUT_WEIGHTS}" \
    --background_trees "${BACKGROUND_TREES[@]}" \
    --signal_trees "${SIGNAL_TREES[@]}" \
    --max_iter 10000 \
    --batch_size 128 \
    --lr 0.001 \
    --num_workers 0 \
    --val_freq 10 \
    --stat_freq 1

echo "Training script finished. Model weights saved to ${OUTPUT_WEIGHTS}"
