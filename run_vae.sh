#!/bin/bash

# --- Environment Setup ---
# Activate your conda environment
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env

# --- Configuration ---
# Define the path to your data and where to save the model
ROOT_FILE="./filtered_numu_cc_events.root"
OUTPUT_WEIGHTS="./test_vae_weights.pth"

# Define the TTree names that contain your background-only training data
TREE_NAMES=(
    "mc_inclusive_run1_fhc"
)

# --- Execute Training ---
# This command starts the training process.
# The new script includes a --kld_weight argument, which you can add if needed.
echo "Starting VAE training..."
python train_vae.py \
    --test \
    --root_file_path "${ROOT_FILE}" \
    --weights "${OUTPUT_WEIGHTS}" \
    --max_iter 1250 \
    --batch_size 32 \
    --kl_weight 1e-3 \
    --lr 0.001 \
    --num_workers 0 \
    --val_freq 10 \
    --stat_freq 1

echo "Training script finished. Model weights saved to ${OUTPUT_WEIGHTS}"
