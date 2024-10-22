#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CURRENT_DIR=$(pwd)
CURRENT_USER=$(whoami)

export NUM_EVENTS=10000
export IMAGE_SIZE="256 256"
export FILE_PREFIX="training_output_"
export BATCH_SIZE=32
export NUM_CLASSES=4
export N_EPOCHS=50
export MODEL_NAME="model"
export VIEW="U"
export VERTEX_PASS=1
export SEED=12345

setup_environment() {
    case $1 in
        default)
            echo -e "${BLUE}-- Setting up default environment...${NC}"
            export DATA_DIR="/gluster/data/dune/cthorpe/kaon_dl"
            export RAW_DIR="$DATA_DIR/make_k0signal_overlay_testing_reco2_reco2"
            export PROCESSED_DIR="$DATA_DIR/processed"
            export OUTPUT_DIR="$DATA_DIR"
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
            export SERVER="uboonegpvm04.fnal.gov"
            export REMOTE_DIR="/exp/uboone/app/users/nlane/production/KaonShortProduction04/srcs/ubana/ubana/searchingforstrangeness/"
            export LOCAL_DIR="$DATA_DIR"
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
echo "RAW_DIR: $RAW_DIR"
echo "PROCESSED_DIR: $PROCESSED_DIR"
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
