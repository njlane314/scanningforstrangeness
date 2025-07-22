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
PYTHON_SCRIPT="train_tripleresnet.py"
# Path where the trained model weights will be saved.
OUTPUT_WEIGHTS="./triple_resnet_model.pth"

# --- Data Configuration ---
# A list of TTree names within the ROOT file to process.
# CORRECTED: Using the specified signal and background TTrees.
TREE_NAMES=("mc_strangeness_run1_fhc" "mc_inclusive_run1_fhc")
# The three branch names for the U, V, and W plane images.
IMAGE_BRANCHES=("detector_image_u" "detector_image_v" "detector_image_w")
# The branch name for event labels.
LABEL_BRANCH="inclusive_strange_channels"
# The branch name for event weights.
WEIGHT_BRANCH="base_event_weight"

# --- Training Hyperparameters ---
LEARNING_RATE=0.00001
BATCH_SIZE=16
MAX_STEPS=10000
VAL_FREQUENCY=100
NUM_WORKERS=0
SEED=777


# ==============================================================================
# ## --- EXECUTION BLOCK --- ##
# ==============================================================================
echo "Starting Triple ResNet classification training..."

# The 'python' command calls the training script.
# Each line with a backslash (\) is a continuation of the same command.
# We pass the bash variables defined above as arguments to the script.
# Note: "${TREE_NAMES[@]}" is the correct way to pass all elements of a bash array.
python "${PYTHON_SCRIPT}" \
    --root_file_path "${ROOT_FILE}" \
    --image_branches "${IMAGE_BRANCHES[@]}" \

echo "Training complete. Model weights saved to ${OUTPUT_WEIGHTS}"
