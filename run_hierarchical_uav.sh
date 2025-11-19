#!/bin/bash

# Hierarchical UAV System Evaluation Script
# Runs H-UAV and L-UAV on separate GPUs

# Configuration
MODEL_PATH="./models/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TEST_DATA="./data/metadata/airspatial_agent_test_task1.jsonl"
OUTPUT_DIR="./outputs/hierarchical_uav"
PORT=50051

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Hierarchical UAV System - Task 1 Evaluation"
echo "=========================================="
echo ""

# Check if test data exists
if [ ! -f "$TEST_DATA" ]; then
    echo "✗ Test data not found: $TEST_DATA"
    echo "  Please run ./setup_data.sh first"
    exit 1
fi

# Function to start H-UAV
start_huav() {
    echo "Starting H-UAV (GPU 0)..."
    echo "  Model: $MODEL_PATH"
    echo "  Vision Tower: $VISION_TOWER"
    echo "  Port: $PORT"
    echo ""

    python hierarchical_uav/eval_task1.py \
        --uav_type h-uav \
        --model_path $MODEL_PATH \
        --vision_tower $VISION_TOWER \
        --device cuda:0 \
        --port $PORT \
        --test_data $TEST_DATA \
        --output $OUTPUT_DIR/huav_results.jsonl \
        2>&1 | tee $OUTPUT_DIR/huav.log
}

# Function to start L-UAV
start_luav() {
    echo "Starting L-UAV (GPU 1)..."
    echo "  Model: $MODEL_PATH"
    echo "  Vision Tower: $VISION_TOWER"
    echo "  H-UAV Address: localhost:$PORT"
    echo "  Threshold: 0.7"
    echo ""

    # Wait for H-UAV to be ready
    echo "Waiting for H-UAV to be ready..."
    sleep 5

    # Check if H-UAV is reachable
    nc -zv localhost $PORT 2>&1 > /dev/null
    if [ $? -ne 0 ]; then
        echo "✗ H-UAV is not reachable at localhost:$PORT"
        echo "  Please ensure H-UAV is running first"
        exit 1
    fi

    echo "✓ H-UAV is reachable"
    echo ""

    python hierarchical_uav/eval_task1.py \
        --uav_type l-uav \
        --model_path $MODEL_PATH \
        --vision_tower $VISION_TOWER \
        --device cuda:1 \
        --huav_address localhost:$PORT \
        --threshold 0.7 \
        --test_data $TEST_DATA \
        --output $OUTPUT_DIR/luav_results.jsonl \
        2>&1 | tee $OUTPUT_DIR/luav.log
}

# Parse command line argument
if [ "$1" == "h-uav" ]; then
    start_huav
elif [ "$1" == "l-uav" ]; then
    start_luav
elif [ "$1" == "both" ]; then
    echo "Running both H-UAV and L-UAV..."
    echo ""

    # Start H-UAV in background
    start_huav &
    HUAV_PID=$!

    # Wait a bit for H-UAV to initialize
    sleep 10

    # Start L-UAV
    start_luav

    # Wait for L-UAV to finish
    echo ""
    echo "L-UAV evaluation completed"
    echo "Shutting down H-UAV..."

    # Stop H-UAV
    kill $HUAV_PID 2>/dev/null
    wait $HUAV_PID 2>/dev/null

    echo ""
    echo "=========================================="
    echo "Evaluation Complete"
    echo "=========================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo "  - huav_results.jsonl"
    echo "  - luav_results.jsonl"
    echo "  - huav.log"
    echo "  - luav.log"
else
    echo "Usage: $0 <h-uav|l-uav|both>"
    echo ""
    echo "Examples:"
    echo "  # Start H-UAV only (server mode)"
    echo "  $0 h-uav"
    echo ""
    echo "  # Start L-UAV only (client mode, requires H-UAV running)"
    echo "  $0 l-uav"
    echo ""
    echo "  # Run both sequentially (automated)"
    echo "  $0 both"
    echo ""
    echo "Manual mode (recommended for debugging):"
    echo "  # Terminal 1 - Start H-UAV"
    echo "  $0 h-uav"
    echo ""
    echo "  # Terminal 2 - Start L-UAV (after H-UAV is ready)"
    echo "  $0 l-uav"
    exit 1
fi
