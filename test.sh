#!/bin/bash

python train_sparse.py \
    --root_file_path "./filtered_numu_cc_events.root" \
    --tree_names "mc_inclusive_run1_fhc" \
    --image_branch "detector_image_w" \
    --label_branch "inclusive_strange_channels" \
    --weight_branch "base_event_weight" \
    --batch_size 128 \
    --lr 0.01 \
    --max_steps 10000 \
    --val_freq 10000 \
    --num_workers 0 \
    --stat_freq 1 \
    --weights "neutrino_classifier_w_plane.pth"

echo "Training script finished."
