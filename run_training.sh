#!/bin/bash

for subdir in models stats images; do
    dir="${OUTPUT_RESULTS_DIR}/${subdir}/pass${ENABLE_VERTEX_PASS}/${DETECTOR_VIEW}"
    [ ! -d "$dir" ] && mkdir -p "$dir"
done

echo -e "${BLUE}-- Running training with PROCESSED_DATA_DIR=$PROCESSED_DATA_DIR and OUTPUT_RESULTS_DIR=$OUTPUT_RESULTS_DIR${NC}"

python3 src/main.py \
    -i "$PROCESSED_DATA_DIR" \
    -b "$BATCH_SIZE" \
    -n "$NUM_OUTPUT_CLASSES" \
    -e "$NUM_EPOCHS" \
    -m "$MODEL_NAME" \
    -v "$DETECTOR_VIEW" \
    -p "$ENABLE_VERTEX_PASS" \
    -s "$RANDOM_SEED" \
    -o "$OUTPUT_RESULTS_DIR"

[ $? -eq 0 ] && echo -e "${GREEN}-- Training completed successfully.${NC}" || (echo -e "${RED}-- Training failed.${NC}" && exit 1)
