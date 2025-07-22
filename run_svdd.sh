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
PYTHON_SCRIPT="train_svdd.py"
OUTPUT_WEIGHTS="./deep_svdd_net.pth"
LEARNING_RATE=0.001
BATCH_SIZE=32
MAX_ITERATIONS=5000
OUTPUT_FEATURES=128
VALIDATION_FREQUENCY=20
TEST_SAMPLE_SIZE=640


# ==============================================================================
# ## --- EXECUTION BLOCK --- ##
# ==============================================================================
echo "Starting Deep SVDD training..."
python "${PYTHON_SCRIPT}" \
    --root_file_path "${ROOT_FILE}" \
    --weights "${OUTPUT_WEIGHTS}" \
    --lr ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --max_iter ${MAX_ITERATIONS} \
    --out_features ${OUTPUT_FEATURES} \
    --val_freq ${VALIDATION_FREQUENCY} \
    --test_sample_size ${TEST_SAMPLE_SIZE}

echo "Training complete. Model weights saved to ${OUTPUT_WEIGHTS}"
