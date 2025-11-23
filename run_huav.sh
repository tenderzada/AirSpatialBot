#!/bin/bash

# H-UAV Startup Script
# Memory Injection Implementation - High Resource UAV

echo "============================================================"
echo "Starting H-UAV (High Resource UAV)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TEST_DATA="/mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/hierarchical_uav"
PORT=50051

# Load trained memory (optional)
# Set LOAD_MEMORY environment variable to load trained MAC memory
# Example: LOAD_MEMORY=./outputs/huav_training/huav_memory_final.pt ./run_huav.sh
LOAD_MEMORY=${LOAD_MEMORY:-""}

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Vision Tower: $VISION_TOWER"
echo "  Device: cuda:0"
echo "  Port: $PORT"
if [ -n "$LOAD_MEMORY" ]; then
    echo "  Trained Memory: $LOAD_MEMORY"
fi
echo "  Output: $OUTPUT_DIR/task1_h-uav_results.jsonl"
echo ""

# Run H-UAV
LOAD_MEMORY_ARG=""
if [ -n "$LOAD_MEMORY" ]; then
    LOAD_MEMORY_ARG="--load-memory $LOAD_MEMORY"
fi

python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cuda:0 \
    --port $PORT \
    --model_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --test_data $TEST_DATA \
    --image_dir $IMAGE_DIR \
    --load_8bit \
    $LOAD_MEMORY_ARG \
    --output $OUTPUT_DIR/task1_h-uav_results.jsonl

echo ""
echo "============================================================"
echo "H-UAV Completed"
echo "============================================================"
echo "Memory saved to: $OUTPUT_DIR/task1_h-uav_results_memory.pt"
echo "Results saved to: $OUTPUT_DIR/task1_h-uav_results.jsonl"
