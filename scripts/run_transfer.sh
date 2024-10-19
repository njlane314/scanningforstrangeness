#!/bin/bash

USERNAME="nlane"
SERVER="uboonegpvm04.fnal.gov"
REMOTE_DIR="/exp/uboone/app/users/nlane/production/KaonShortProduction04/srcs/ubana/ubana/searchingforstrangeness/"
LOCAL_DIR="/Users/user/conv_network/scanningforstrangeness/data/raw"

FILES=("training_output_U.csv" "training_output_V.csv" "training_output_W.csv")

echo "Removing previous training output samples..."
rm -f ${LOCAL_DIR}training_output_*.csv

for FILE in "${FILES[@]}"; do
    echo "Transferring ${FILE}..."
    scp ${USERNAME}@${SERVER}:${REMOTE_DIR}${FILE} ${LOCAL_DIR}

    if [ $? -eq 0 ]; then
        echo "Successfully transferred ${FILE}."
    else
        echo "Failed to transfer ${FILE}."
    fi
done

echo "All transfers complete."

