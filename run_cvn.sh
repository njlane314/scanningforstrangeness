#!/bin/bash

# --- Environment Setup ---
# This section should match the setup for your specific computing environment.
echo "Setting up the environment..."
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env

# --- File and Data Configuration ---
# Path to the Python script that contains the model and training loop.
PYTHON_SCRIPT="train_dunecvn.py" 

# Path to the input ROOT file containing the event data.
ROOT_FILE="./filtered_numu_cc_events.root" 

# Comma-separated list of TTree names to process within the ROOT file.
TREE_NAMES="mc_inclusive_run1_fhc,mc_strangeness_run1_fhc"

# Path where the best model weights will be saved.
OUTPUT_WEIGHTS="./dunecvn_model.pth"

# --- Training Hyperparameters (from Phys. Rev. D 102, 092003) ---
LEARNING_RATE=0.01
MOMENTUM=0.9
WEIGHT_DECAY=0.0001
EPOCHS=15
BATCH_SIZE=32
NUM_WORKERS=0 # Adjust based on your machine's core count
SEED=777

# --- Execute the Training Script ---
echo "Starting DUNE CVN training..."
python "${PYTHON_SCRIPT}" \
    --root_file_path "${ROOT_FILE}" \
    --tree_names "${TREE_NAMES}" \
    --lr "${LEARNING_RATE}" \
    --momentum "${MOMENTUM}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --weights "${OUTPUT_WEIGHTS}"

echo "Training complete. Model weights saved to ${OUTPUT_WEIGHTS}"