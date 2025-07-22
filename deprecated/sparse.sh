#!/bin/bash
set -e

echo "================================================="
echo "Job started on node: $(hostname)"
echo "Cluster: $(Cluster), Process: $(Process)"
echo "Executing in: $(pwd)"
echo "================================================="

REMOTE_ROOT_FILE="$1"
REMOTE_OUTPUT_DIR="$2"

echo "Sourcing Conda environment..."
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env

LOCAL_ROOT_FILE="/tmp/local_events_$(Cluster)_$(Process).root"
echo "Copying data from ${REMOTE_ROOT_FILE} to ${LOCAL_ROOT_FILE}..."
cp "${REMOTE_ROOT_FILE}" "${LOCAL_ROOT_FILE}"
echo "Copy complete."

LOCAL_OUTPUT_DIR="model_output"
mkdir -p "${LOCAL_OUTPUT_DIR}"
echo "Local output will be saved to ${LOCAL_OUTPUT_DIR}"

echo "Starting Python script..."
python -u train_sparse.py \
    --root-file "${LOCAL_ROOT_FILE}" \
    --output-dir "${LOCAL_OUTPUT_DIR}"
echo "Python script finished."

echo "Cleaning up local data file: ${LOCAL_ROOT_FILE}"
rm "${LOCAL_ROOT_FILE}"

echo "Job finished successfully."