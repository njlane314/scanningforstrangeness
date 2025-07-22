#!/bin/bash

# ==============================================================================
# ## --- Environment Setup --- ##
# ==============================================================================
echo "Activating Conda environment..."
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env


# ==============================================================================
# ## --- Configuration --- ##
# ==============================================================================
ROOT_FILE="./filtered_numu_cc_events.root"
PYTHON_SCRIPT="train_binary.py"
OUTPUT_WEIGHTS="./binary_classifier_resnet34.pth"

# --- Training Hyperparameters ---
LEARNING_RATE=0.0001
BATCH_SIZE=32
MAX_ITERATIONS=10000
VAL_FREQUENCY=100
TEST_SAMPLE_SIZE=1000


# ==============================================================================
# ## --- EXECUTION BLOCK --- ##
# ==============================================================================
echo "Starting binary classification training..."
python "${PYTHON_SCRIPT}" \
    --root_file_path "${ROOT_FILE}" \
    --weights "${OUTPUT_WEIGHTS}" \
    --lr ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --max_iter ${MAX_ITERATIONS} \
    --val_freq ${VAL_FREQUENCY} \
    --test_sample_size ${TEST_SAMPLE_SIZE}

echo "Training complete. Model weights saved to ${OUTPUT_WEIGHTS}"