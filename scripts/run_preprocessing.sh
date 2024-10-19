#!/bin/bash

source scripts/config_env.sh

if [ ! -d "$PROCESSED_DIR" ]; then
    mkdir -p "$PROCESSED_DIR"
fi

echo -e "${BLUE}-- Running preprocessing with RAW_DIR=$RAW_DIR and PROCESSED_DIR=$PROCESSED_DIR${NC}"

python3 src/data/process_data.py \
    --raw_dir "$RAW_DIR" \
    --processed_dir "$PROCESSED_DIR" \
    --file_prefix "$FILE_PREFIX" \
    --image_size $IMAGE_SIZE

if [ $? -eq 0 ]; then
    echo -e "${GREEN}-- Preprocessing completed successfully.${NC}"
else
    echo -e "${RED}-- Preprocessing failed.${NC}"
    exit 1
fi

echo -e "${YELLOW}-- Processed data saved to $PROCESSED_DIR${NC}"