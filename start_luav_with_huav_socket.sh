#!/bin/bash

# Start L-UAV with H-UAV Collaboration (Socket-based)
#
# 需要先启动 H-UAV server:
#   ./start_huav_lora_socket.sh

echo "============================================================"
echo "L-UAV + H-UAV Collaboration Evaluation (Socket-based)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TEST_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/eval_luav_huav_socket"

# L-UAV device (different from H-UAV)
LUAV_DEVICE="cuda:1"

# H-UAV connection
HUAV_ADDRESS="localhost:50051"
CONFIDENCE_THRESHOLD=0.5

# Testing: Force H-UAV query every N samples (for verifying collaboration)
FORCE_HUAV_EVERY_N=5  # Set to empty "" to disable

# Optional: Limit samples for testing
MAX_SAMPLES=""  # Empty = all samples

echo ""
echo "Configuration:"
echo "  L-UAV model: $MODEL_PATH"
echo "  L-UAV device: $LUAV_DEVICE"
echo "  Vision tower: $VISION_TOWER"
echo "  H-UAV address: $HUAV_ADDRESS"
echo "  Confidence threshold: $CONFIDENCE_THRESHOLD"
if [ -n "$FORCE_HUAV_EVERY_N" ]; then
    echo "  🔧 Force H-UAV query: Every $FORCE_HUAV_EVERY_N samples"
fi
echo "  Test data: $TEST_DATA"
echo "  Image dir: $IMAGE_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Collaboration Strategy:"
echo "  - L-UAV performs inference first"
echo "  - If confidence < $CONFIDENCE_THRESHOLD, request H-UAV memory"
if [ -n "$FORCE_HUAV_EVERY_N" ]; then
    echo "  - 🔧 Also force query every $FORCE_HUAV_EVERY_N samples (testing mode)"
fi
echo "  - H-UAV provides LoRA-enhanced memory tokens"
echo "  - L-UAV performs enhanced inference"
echo ""
echo "============================================================"

# Check H-UAV connectivity
echo ""
echo "Checking H-UAV server connectivity..."
if timeout 2 bash -c "echo > /dev/tcp/localhost/50051" 2>/dev/null; then
    echo "✓ H-UAV server is reachable at $HUAV_ADDRESS"
else
    echo ""
    echo "❌ H-UAV server not reachable at $HUAV_ADDRESS"
    echo ""
    echo "Please start H-UAV server first:"
    echo "  ./start_huav_lora_socket.sh"
    echo ""
    exit 1
fi

echo ""
echo "Starting L-UAV + H-UAV evaluation..."
echo ""

# Build command
CMD="python hierarchical_uav/eval_sqa_lora_socket.py \
    --model_path \"$MODEL_PATH\" \
    --vision_tower \"$VISION_TOWER\" \
    --device \"$LUAV_DEVICE\" \
    --huav_address \"$HUAV_ADDRESS\" \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --test_data \"$TEST_DATA\" \
    --image_dir \"$IMAGE_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --load_8bit"

# Add max_samples if set
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# Add force_huav_every_n if set
if [ -n "$FORCE_HUAV_EVERY_N" ]; then
    CMD="$CMD --force_huav_every_n $FORCE_HUAV_EVERY_N"
fi

# Execute
eval $CMD

EVAL_EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "L-UAV + H-UAV evaluation completed successfully"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/results.json"
else
    echo "L-UAV + H-UAV evaluation exited with error (code: $EVAL_EXIT_CODE)"
fi
echo "============================================================"
