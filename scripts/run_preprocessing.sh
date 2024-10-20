#!/bin/bash

source scripts/config_env.sh

[ ! -d "$PROCESSED_DIR" ] && mkdir -p "$PROCESSED_DIR"

echo -e "${BLUE}-- Running preprocessing with RAW_DIR=$RAW_DIR and PROCESSED_DIR=$PROCESSED_DIR${NC}"

python3 src/data/process_data.py \
    -r "$RAW_DIR" \
    -p "$PROCESSED_DIR" \
    -f "$FILE_PREFIX" \
    -s $IMAGE_SIZE

[ $? -eq 0 ] && echo -e "${GREEN}-- Preprocessing completed successfully.${NC}" || (echo -e "${RED}-- Preprocessing failed.${NC}" && exit 1)

echo -e "${YELLOW}-- Processed data saved to $PROCESSED_DIR${NC}"