#!/bin/bash

# ==============================================================================
# ## --- BASH SCRIPT FOR VISUALIZING THE LEARNED VAE CONCEPT --- ##
# ==============================================================================

# --- Environment Setup ---
# This line activates your specific conda environment.
# Make sure the path is correct for your system.
echo "Activating conda environment..."
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env

# --- Configuration ---
# Define the paths to your data file and the trained model weights.
# These should match the files used in your training run.
ROOT_FILE="./filtered_numu_cc_events.root"
TREE_NAMES="mc_inclusive_run1_fhc"
SAVED_WEIGHTS="./test_vae_weights.pth" # The weights file saved by the training script

# --- Check for Weights File ---
# This check ensures the script doesn't run if the model hasn't been trained yet.
if [ ! -f "$SAVED_WEIGHTS" ]; then
    echo "Error: Weights file not found at ${SAVED_WEIGHTS}"
    echo "Please train the model first using your training script."
    exit 1
fi

# --- Execute Visualization Script ---
# This command runs the visualization script, passing the necessary file paths
# as command-line arguments.
echo "Starting visualization of the learned concept..."
python visualise.py \
    --root_file_path "${ROOT_FILE}" \
    --tree_names "${TREE_NAMES}" \
    --weights "${SAVED_WEIGHTS}" \
    --batch_size 64 \
    --num_workers 0

echo "-----------------------------------------------------"
echo "Visualization complete."
echo "Check for the output image: average_background_visualization.png"
echo "-----------------------------------------------------"