#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CURRENT_DIR=$(pwd)

if [[ "$HOSTNAME" == *"cluster"* ]]; then
    ENVIRONMENT="cluster"
else
    ENVIRONMENT="local"
fi

if [ "$ENVIRONMENT" = "cluster" ]; then
    echo -e "${BLUE}-- Setting up paths for cluster environment...${NC}"
    export RAW_DIR="$CURRENT_DIR/data/raw"
    export PROCESSED_DIR="$CURRENT_DIR/data/processed"
    export IMAGE_PATH="$CURRENT_DIR/images"
    export OUTPUT_DIR="$CURRENT_DIR/outputs"
else
    echo -e "${YELLOW}-- Setting up paths for local environment...${NC}"
    export RAW_DIR="$CURRENT_DIR/data/raw"
    export PROCESSED_DIR="$CURRENT_DIR/data/processed"
    export IMAGE_PATH="$CURRENT_DIR/images"
    export OUTPUT_DIR="$CURRENT_DIR/outputs"
fi