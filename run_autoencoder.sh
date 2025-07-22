#!/bin/bash

# ==============================================================================
# ## --- Environment Setup --- ##
# ==============================================================================
# This section activates the correct Conda environment for the training script.
# IMPORTANT: Update the path to your Conda installation if it differs.
echo "Activating Conda environment..."
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env


# ==============================================================================
# ## --- Configuration --- ##
# ==============================================================================
# --- File and Script Paths ---
# Path to the input ROOT file containing the event data.
ROOT_FILE="./filtered_numu_cc_events.root"
# The name of the Python script to execute.
PYTHON_SCRIPT="train_autoencoder.py"
# Path where the trained model weights will be saved.
OUTPUT_WEIGHTS="./triple_autoencoder_model.pth"

# --- Data Configuration ---
# A list of TTree names within the ROOT file to process.
TREE_NAMES=("mc_strangeness_run1_fhc" "mc_inclusive_run1_fhc")
# The three branch names for the U, V, and W plane images.
IMAGE_BRANCHES=("detector_image_u" "detector_image_v" "detector_image_w")
# The branch name for event labels (used for splitting and final evaluation).
LABEL_BRANCH="inclusive_strange_channels"
# The branch name for event weights (not used in training, but required by the script).
WEIGHT_BRANCH="base_event_weight"

# --- Training Hyperparameters ---
LEARNING_RATE=0.0001
BATCH_SIZE=32
MAX_STEPS=10000
# How many events to use for the final AUC calculation.
TEST_SPLIT_SIZE=1000
NUM_WORKERS=0
SEED=777


# ==============================================================================
# ## --- EXECUTION BLOCK --- ##
# ==============================================================================
echo "Starting Triple Autoencoder training for anomaly detection..."

# The 'python' command calls the training script.
# Each line with a backslash (\) is a continuation of the same command.
# We pass the bash variables defined above as arguments to the script.
python "${PYTHON_SCRIPT}" \
    --root_file_path "${ROOT_FILE}" \
    --tree_names "${TREE_NAMES[@]}" \
    --image_branches "${IMAGE_BRANCHES[@]}" \
    --label_branch "${LABEL_BRANCH}" \
    --weight_branch "${WEIGHT_BRANCH}" \
    --weights "${OUTPUT_WEIGHTS}" \
    --lr ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --max_steps ${MAX_STEPS} \
    --test_split_size ${TEST_SPLIT_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}

echo "Training complete. Model weights saved to ${OUTPUT_WEIGHTS}"