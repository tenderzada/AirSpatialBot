#!/bin/bash

# L-UAV Remote Startup Script
# Connects to H-UAV on remote machine (10.37.74.206:50051)

echo "============================================================"
echo "Starting L-UAV (Remote Client Mode)"
echo "============================================================"

# Configuration - ADJUST THESE PATHS ON REMOTE MACHINE
MODEL_PATH="/mnt/data/AirSpatialBot"  # ← 修改为远程机器的路径
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"  # ← 修改为远程机器的路径
TEST_DATA="/mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl"  # ← 修改为远程机器的路径
IMAGE_DIR="/mnt/data/AirSpatial/images"  # ← 修改为远程机器的路径
OUTPUT_DIR="./outputs/hierarchical_uav"

# Remote H-UAV Configuration
HUAV_ADDRESS="10.37.74.206:50051"  # H-UAV server address
THRESHOLD=1.0  # Force query H-UAV for memory testing
FORCE_QUERY_RATE=0.3  # 30% samples bypass KB to test memory

# Device configuration - adjust based on remote machine GPU
DEVICE="cuda:0"  # ← 修改为远程机器的 GPU

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Vision Tower: $VISION_TOWER"
echo "  Device: $DEVICE"
echo "  H-UAV Address: $HUAV_ADDRESS"
echo "  Self-matching Threshold: $THRESHOLD"
echo "  Force Query Rate: $FORCE_QUERY_RATE"
echo "  Output: $OUTPUT_DIR/task1_l-uav_remote_results.jsonl"
echo ""

# Test H-UAV connection
echo "Testing H-UAV connection..."
if nc -z 10.37.74.206 50051 2>/dev/null; then
    echo "✓ H-UAV is reachable at $HUAV_ADDRESS"
else
    echo "⚠ Warning: Cannot connect to H-UAV at $HUAV_ADDRESS"
    echo "  Please ensure:"
    echo "  1. H-UAV server is running on 10.37.74.206"
    echo "  2. Port 50051 is open in firewall"
    echo "  3. Network connectivity is stable"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo ""
echo "Starting L-UAV remote evaluation..."
echo ""

# Run L-UAV connecting to remote H-UAV
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device $DEVICE \
    --huav_address $HUAV_ADDRESS \
    --threshold $THRESHOLD \
    --force-query-rate $FORCE_QUERY_RATE \
    --model_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --test_data $TEST_DATA \
    --image_dir $IMAGE_DIR \
    --load_8bit \
    --output $OUTPUT_DIR/task1_l-uav_remote_results.jsonl

echo ""
echo "============================================================"
echo "L-UAV Remote Evaluation Completed"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR/task1_l-uav_remote_results.jsonl"
echo ""
echo "To analyze results:"
echo "  python analyze_uav_results.py $OUTPUT_DIR/task1_l-uav_remote_results.jsonl"
echo ""
echo "To compare with baseline:"
echo "  python compare_baseline_vs_memory.py \\"
echo "    outputs/hierarchical_uav/task1_l-uav_standalone.jsonl \\"
echo "    outputs/hierarchical_uav/task1_l-uav_remote_results.jsonl"
echo ""
