#!/bin/bash

[ ! -d "$PROCESSED_DATA_DIR" ] && mkdir -p "$PROCESSED_DATA_DIR"

LOG_FILE="$OUTPUT_RESULTS_DIR/preprocessing_log.txt"
mkdir -p "$OUTPUT_RESULTS_DIR"

{
    echo -e "${BLUE}-- Running preprocessing with RAW_DATA_PATH=$RAW_DATA_PATH and PROCESSED_DATA_DIR=$PROCESSED_DATA_DIR${NC}"

    python3 src/imager.py \
        -r "$RAW_DATA_PATH" \
        -p "$PROCESSED_DATA_DIR" \
        -f "$OUTPUT_FILE_PREFIX" \
        -s $IMAGE_DIMENSIONS \
        -n $NUM_EVENTS

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}-- Preprocessing completed successfully.${NC}"
    else
        echo -e "${RED}-- Preprocessing failed.${NC}"
        exit 1
    fi

    echo -e "${CYAN}-- Processed data saved to $PROCESSED_DATA_DIR${NC}"

} | tee "$LOG_FILE"
