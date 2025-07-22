#!/bin/bash

# 1. Activate the Conda environment
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env

# 2. Define file paths
ROOT_FILE="./filtered_numu_cc_events.root"
INPUT_WEIGHTS="./vae_weights.pth"
TREE_NAMES=(
    "mc_inclusive_run1_fhc"
)

# 3. Choose which events to visualize
#    - Add all desired event indices to this space-separated list.
EVENT_INDICES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

# 4. Define the output file name pattern
#    - The python script will replace '{index}' with the actual event index.
OUTPUT_PATTERN="./reco_event_{index}.png"

# Create the output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_PATTERN")"

# 5. Run the visualization script ONCE with all indices
echo "Running visualization for event indices: ${EVENT_INDICES[*]}..."
python visualise_vae.py \
    --root_file_path "${ROOT_FILE}" \
    --tree_names "${TREE_NAMES[@]}" \
    --weights "${INPUT_WEIGHTS}" \
    --indices "${EVENT_INDICES[@]}" \
    --output_pattern "${OUTPUT_PATTERN}"

echo "All visualization scripts finished."