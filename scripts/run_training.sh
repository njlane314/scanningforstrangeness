#!/bin/bash

source scripts/config_env.sh

for subdir in models stats images; do
    dir="${OUTPUT_DIR}/${subdir}/pass${VERTEX_PASS}/${VIEW}"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
    fi
done

echo -e "${BLUE}-- Running training with IMAGE_PATH=$IMAGE_PATH and OUTPUT_DIR=$OUTPUT_DIR${NC}"

python3 src/main.py \
    --image_path "$IMAGE_PATH" \
    --batch_size "$BATCH_SIZE" \
    --num_classes "$NUM_CLASSES" \
    --n_epochs "$N_EPOCHS" \
    --model_name "$MODEL_NAME" \
    --view "$VIEW" \
    --vertex_pass "$VERTEX_PASS" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}-- Training completed successfully.${NC}"
else
    echo -e "${RED}-- Training failed.${NC}"
    exit 1
fi
