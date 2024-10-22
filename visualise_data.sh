#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo -e "${RED}Usage: $0 <view> <event_number> | --all${NC}"
    exit 1
fi

VIEW=$1

if [ ! -d "$PROCESSED_DATA_DIR" ]; then
    echo -e "${RED}Error: Processed directory $PROCESSED_DATA_DIR does not exist.${NC}"
    exit 1
fi

if [ "$2" == "--all" ]; then
    echo -e "${BLUE}-- Running visualisation for all events in view=${VIEW}${NC}"

    input_folder="$PROCESSED_DATA_DIR/images_${VIEW}/"
    if [ ! -d "$input_folder" ]; then
        echo -e "${RED}Error: Input folder for view $VIEW not found.${NC}"
        exit 1
    fi

    for input_file in "$input_folder"/image_*.npz; do
        event_number=$(basename "$input_file" | sed 's/^image_//;s/\.npz$//')
        echo -e "${CYAN}-- Visualising event ${event_number}${NC}"

        python3 src/visualise/visualise_data.py \
            -p "$PROCESSED_DATA_DIR" \
            -v "$VIEW" \
            -e "$event_number"

        if [ $? -ne 0 ]; then
            echo -e "${RED}-- Visualisation failed for event ${event_number}.${NC}"
        fi
    done

    echo -e "${GREEN}-- Visualisation for all events completed.${NC}"
else
    EVENT_NUMBER=$2
    echo -e "${BLUE}-- Running visualisation for view=${VIEW}, event_number=${EVENT_NUMBER}${NC}"

    python3 src/visualise/visualise_data.py \
        -p "$PROCESSED_DATA_DIR" \
        -v "$VIEW" \
        -e "$EVENT_NUMBER"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}-- Visualisation completed successfully.${NC}"
    else
        echo -e "${RED}-- Visualisation failed.${NC}"
        exit 1
    fi
fi
