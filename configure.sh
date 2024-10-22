#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

WORKING_DIR=$(pwd)
CURRENT_USER=$(whoami)

export NUM_EVENTS=10000
export RAW_DATA_SUBDIR="make_k0signal_overlay_testing_reco2_reco2"

export IMAGE_DIMENSIONS="256 256"
export OUTPUT_FILE_PREFIX="training_output_"
export BATCH_SIZE=32
export NUM_OUTPUT_CLASSES=4
export NUM_EPOCHS=50
export MODEL_NAME="testing_model"
export DETECTOR_VIEW="U"
export ENABLE_VERTEX_PASS=1
export RANDOM_SEED=12345

setup_environment() {
    case $1 in
        default)
            echo -e "${BLUE}-- Setting up default environment...${NC}"
            export RAW_DATA_DIR="/gluster/data/dune/cthorpe/kaon_dl"
            export RAW_DATA_PATH="$RAW_DATA_DIR/$RAW_DATA_SUBDIR"

            export PROCESSED_DATA_DIR="/gluster/data/dune/niclane/scans/$RAW_DATA_SUBDIR"
            export OUTPUT_RESULTS_DIR="$PROCESSED_DATA_DIR/out"
            mkdir -p "$PROCESSED_DATA_DIR"
            mkdir -p "$OUTPUT_RESULTS_DIR"
        ;;
        *)
            echo -e "${RED}-- No matching case found. Falling back to default environment...${NC}"
            setup_environment default
            ;;
    esac
}

setup_user() {
    case $CURRENT_USER in
        niclane)
            echo -e "${GREEN}-- Setting up user environment for nlane...${NC}"
            export USERNAME="nlane"
            export REMOTE_SERVER="uboonegpvm04.fnal.gov"
            export REMOTE_WORK_DIR="/exp/uboone/app/users/nlane/production/KaonShortProduction04/srcs/ubana/ubana/searchingforstrangeness/"
            export LOCAL_WORK_DIR="/gluster/home/$CURRENT_USER"
            ;;
        *)
            echo -e "${RED}-- No matching user found. Falling back to default user configuration...${NC}"
            setup_environment niclane
            ;;
    esac
}

setup_environment $1
setup_user

echo -e "${BLUE}-- Environment configuration:${NC}"
echo "RAW_DATA_PATH: $RAW_DATA_PATH"
echo "PROCESSED_DATA_DIR: $PROCESSED_DATA_DIR"
echo "OUTPUT_RESULTS_DIR: $OUTPUT_RESULTS_DIR"
echo "IMAGE_DIMENSIONS: $IMAGE_DIMENSIONS"
echo "OUTPUT_FILE_PREFIX: $OUTPUT_FILE_PREFIX"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "NUM_OUTPUT_CLASSES: $NUM_OUTPUT_CLASSES"
echo "NUM_EPOCHS: $NUM_EPOCHS"
echo "MODEL_NAME: $MODEL_NAME"
echo "DETECTOR_VIEW: $DETECTOR_VIEW"
echo "ENABLE_VERTEX_PASS: $ENABLE_VERTEX_PASS"
echo "RANDOM_SEED: $RANDOM_SEED"

echo -e "${GREEN}-- User configuration:${NC}"
echo "USERNAME: $USERNAME"
echo "REMOTE_SERVER: $REMOTE_SERVER"
echo "REMOTE_WORK_DIR: $REMOTE_WORK_DIR"
echo "LOCAL_WORK_DIR: $LOCAL_WORK_DIR"

alias cdd="cd $RAW_DATA_DIR"
alias cdw="cd $WORKING_DIR"

echo -e "${CYAN}-- Available aliases:${NC}"
alias | grep 'cd'
