#!/bin/bash

for subdir in models stats images; do
    dir="${OUTPUT_DIR}/${subdir}/pass${VERTEX_PASS}/${VIEW}"
    [ ! -d "$dir" ] && mkdir -p "$dir"
done

echo -e "${BLUE}-- Running training with PROCESSED_DIR=$PROCESSED_DIR and OUTPUT_DIR=$OUTPUT_DIR${NC}"

python3 src/main.py \
    -i "$PROCESSED_DIR" \
    -b "$BATCH_SIZE" \
    -n "$NUM_CLASSES" \
    -e "$N_EPOCHS" \
    -m "$MODEL_NAME" \
    -v "$VIEW" \
    -p "$VERTEX_PASS" \
    -s "$SEED" \
    -o "$OUTPUT_DIR"

[ $? -eq 0 ] && echo -e "${GREEN}-- Training completed successfully.${NC}" || (echo -e "${RED}-- Training failed.${NC}" && exit 1)