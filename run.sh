#!/bin/bash

# This script runs the contrastive learning training process.
# It trains a feature extractor on background-only events, using different
# detector planes as natural augmentations. The resulting model is saved
# to be used for anomaly detection in a separate evaluation step.

source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env

python train_contrastive.py \
    --root_file_path "./filtered_numu_cc_events.root" \
    --tree_names "mc_inclusive_run1_fhc" \
    --image_branches "detector_image_w" "detector_image_u" "detector_image_v" \
    --label_branch "inclusive_strange_channels" \
    --batch_size 128 \
    --lr 10.0 \
    --max_steps 2000 \
    --val_freq 1000 \
    --num_workers 0 \
    --stat_freq 1 \
    --weights "contrastive_feature_extractor.pth" \
    --out_dim 128 \
    --temperature 0.07

echo "Training script finished."