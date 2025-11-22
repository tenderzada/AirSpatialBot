#!/bin/bash

# L-UAV Startup Script
# Memory Injection Implementation - Low Resource UAV

echo "============================================================"
echo "Starting L-UAV (Low Resource UAV)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TEST_DATA="/mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/hierarchical_uav"
HUAV_ADDRESS="localhost:50051"
THRESHOLD=0.7

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Vision Tower: $VISION_TOWER"
echo "  Device: cuda:1"
echo "  H-UAV Address: $HUAV_ADDRESS"
echo "  Self-matching Threshold: $THRESHOLD"
echo "  Output: $OUTPUT_DIR/task1_l-uav_results.jsonl"
echo ""

# Check if H-UAV is reachable
echo "Checking H-UAV connection..."
if nc -z localhost 50051 2>/dev/null; then
    echo "✓ H-UAV is reachable at $HUAV_ADDRESS"
else
    echo "⚠ Warning: Cannot connect to H-UAV at $HUAV_ADDRESS"
    echo "  Make sure H-UAV is running first!"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo ""
echo "Starting L-UAV evaluation..."
echo ""

# Run L-UAV
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device cuda:1 \
    --huav_address $HUAV_ADDRESS \
    --threshold $THRESHOLD \
    --model_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --test_data $TEST_DATA \
    --image_dir $IMAGE_DIR \
    --load_8bit \
    --output $OUTPUT_DIR/task1_l-uav_results.jsonl

echo ""
echo "============================================================"
echo "L-UAV Completed"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR/task1_l-uav_results.jsonl"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/task1_l-uav_results.jsonl | jq '.'"
