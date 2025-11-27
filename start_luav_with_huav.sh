#!/bin/bash

# Start L-UAV with H-UAV LoRA Memory Collaboration
#
# L-UAV 评估脚本：连接到 H-UAV LoRA 服务器获取记忆协助

echo "============================================================"
echo "L-UAV with H-UAV LoRA Memory Collaboration"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TEST_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/eval_luav_with_huav"
DEVICE="cuda:1"

# H-UAV collaboration settings
HUAV_URL="http://localhost:8000"
CONFIDENCE_THRESHOLD=0.5

# Optional: limit samples for quick testing
MAX_SAMPLES=""  # Leave empty for all samples, or set to a number like 100

echo ""
echo "Configuration:"
echo "  L-UAV model: $MODEL_PATH"
echo "  Vision tower: $VISION_TOWER"
echo "  Test data: $TEST_DATA"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""
echo "H-UAV Collaboration:"
echo "  H-UAV URL: $HUAV_URL"
echo "  Confidence threshold: $CONFIDENCE_THRESHOLD"
echo "  (Will request H-UAV memory when confidence < $CONFIDENCE_THRESHOLD)"
echo ""
echo "Architecture:"
echo "  ┌────────────────────────────┐"
echo "  │  L-UAV (Lightweight)       │"
echo "  │  - Base inference          │"
echo "  │  - Self-matching gating    │"
echo "  └────────────────────────────┘"
echo "           ↓ (low confidence)"
echo "  ┌────────────────────────────┐"
echo "  │  Request memory from H-UAV │"
echo "  │  POST /get_memory          │"
echo "  └────────────────────────────┘"
echo "           ↓"
echo "  ┌────────────────────────────┐"
echo "  │  H-UAV returns memory      │"
echo "  │  tokens: [8, 4096]         │"
echo "  └────────────────────────────┘"
echo "           ↓"
echo "  ┌────────────────────────────┐"
echo "  │  L-UAV enhanced inference  │"
echo "  │  memory || input → answer  │"
echo "  └────────────────────────────┘"
echo ""
echo "============================================================"

# Check H-UAV server
echo ""
echo "Checking H-UAV server..."
if curl -s -f -m 2 "$HUAV_URL/health" > /dev/null 2>&1; then
    echo "✓ H-UAV server is running at $HUAV_URL"
else
    echo "⚠️  Warning: Cannot connect to H-UAV server at $HUAV_URL"
    echo ""
    echo "Please start H-UAV server first:"
    echo "  ./start_huav_lora.sh"
    echo ""
    read -p "Continue anyway? (will run in standalone mode) (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

echo ""
echo "Starting L-UAV evaluation..."
echo ""

# Build command
CMD="python hierarchical_uav/eval_sqa_lora.py \
    --model_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --device $DEVICE \
    --test_data $TEST_DATA \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --huav_url $HUAV_URL \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --load_8bit"

# Add max_samples if set
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# Run evaluation
eval $CMD

EVAL_EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "L-UAV Evaluation Complete ✓"
else
    echo "L-UAV Evaluation Failed ✗ (exit code: $EVAL_EXIT_CODE)"
fi
echo "============================================================"

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Results saved to: $OUTPUT_DIR/results.json"
    echo ""

    # Show quick stats if jq is available
    if command -v jq &> /dev/null; then
        if [ -f "$OUTPUT_DIR/results.json" ]; then
            echo "Quick Statistics:"
            echo "  MAE:  $(jq -r '.metrics.mae' $OUTPUT_DIR/results.json)"
            echo "  RMSE: $(jq -r '.metrics.rmse' $OUTPUT_DIR/results.json)"
            echo "  MRE:  $(jq -r '.metrics.mre' $OUTPUT_DIR/results.json)"
            echo ""
            echo "H-UAV Usage:"
            echo "  Total queries: $(jq -r '.stats.total_queries' $OUTPUT_DIR/results.json)"
            echo "  H-UAV requests: $(jq -r '.stats.huav_requests' $OUTPUT_DIR/results.json)"
            echo "  Standalone: $(jq -r '.stats.standalone_inferences' $OUTPUT_DIR/results.json)"
            echo ""
        fi
    fi

    echo "To view full results:"
    echo "  cat $OUTPUT_DIR/results.json | jq ."
    echo ""
fi
