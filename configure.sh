#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CURRENT_DIR=$(pwd)
CURRENT_USER=$(whoami)

setup_environment() {
    case $1 in
        default)
            echo -e "${YELLOW}-- Setting up default environment...${NC}"
            export RAW_DIR="$CURRENT_DIR/data/raw"
            export PROCESSED_DIR="$CURRENT_DIR/data/processed"
            export IMAGE_PATH="$CURRENT_DIR/data/processed"
            export OUTPUT_DIR="$CURRENT_DIR/output"
            export IMAGE_SIZE="256 256"
            export FILE_PREFIX="training_output_"
            BATCH_SIZE=32
            NUM_CLASSES=4
            N_EPOCHS=50
            MODEL_NAME="model"
            VIEW="U"
            VERTEX_PASS=1
            SEED=12345
            ;;
        custom_case)
            echo -e "${BLUE}-- Setting up custom_case environment...${NC}"
            setup_environment default
            ;;
        *)
            echo -e "${RED}-- No matching case found. Falling back to default environment...${NC}"
            setup_environment default
            ;;
    esac
}

setup_user() {
    case $CURRENT_USER in
        r22473nl)
            echo -e "${GREEN}-- Setting up user environment for nlane...${NC}"
            export USERNAME="nlane"
            export SERVER="uboonegpvm04.fnal.gov"
            export REMOTE_DIR="/exp/uboone/app/users/nlane/production/KaonShortProduction04/srcs/ubana/ubana/searchingforstrangeness/"
            export LOCAL_DIR="$CURRENT_DIR/data/raw"
            ;;
        custom_user)
            echo -e "${BLUE}-- Setting up user environment for custom_user...${NC}"
            ;;
        *)
            echo -e "${RED}-- No matching user found. Falling back to default user configuration...${NC}"
            setup_environment r22473nl
            ;;
    esac
}

setup_environment $1
setup_user

echo -e "${BLUE}-- Environment configuration:${NC}"
echo "RAW_DIR: $RAW_DIR"
echo "PROCESSED_DIR: $PROCESSED_DIR"
echo "IMAGE_PATH: $IMAGE_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "IMAGE_SIZE: $IMAGE_SIZE"
echo "FILE_PREFIX: $FILE_PREFIX"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "NUM_CLASSES: $NUM_CLASSES"
echo "N_EPOCHS: $N_EPOCHS"
echo "MODEL_NAME: $MODEL_NAME"
echo "VIEW: $VIEW"
echo "VERTEX_PASS: $VERTEX_PASS"
echo "SEED: $SEED"

echo -e "${GREEN}-- User configuration:${NC}"
echo "USERNAME: $USERNAME"
echo "SERVER: $SERVER"
echo "REMOTE_DIR: $REMOTE_DIR"
echo "LOCAL_DIR: $LOCAL_DIR"
