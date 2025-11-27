#!/bin/bash

# Start L-UAV Standalone (No H-UAV)
#
# L-UAV 独立评估：不使用 H-UAV 记忆，用于基线对比

echo "============================================================"
echo "L-UAV Standalone Evaluation (No H-UAV Memory)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TEST_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/eval_luav_standalone"
DEVICE="cuda:1"

# Optional: limit samples for quick testing
MAX_SAMPLES=""  # Leave empty for all samples

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Vision tower: $VISION_TOWER"
echo "  Test data: $TEST_DATA"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""
echo "Mode: Standalone (Baseline)"
echo "  - No H-UAV memory"
echo "  - Pure L-UAV inference"
echo "  - Use as baseline for comparison"
echo ""
echo "============================================================"
echo ""

echo "Starting L-UAV standalone evaluation..."
echo ""

# Build command (no huav_url = standalone mode)
CMD="python hierarchical_uav/eval_sqa_lora.py \
    --model_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --device $DEVICE \
    --test_data $TEST_DATA \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
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
    echo "L-UAV Standalone Evaluation Complete ✓"
else
    echo "L-UAV Standalone Evaluation Failed ✗ (exit code: $EVAL_EXIT_CODE)"
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
        fi
    fi

    echo "This is the baseline result."
    echo "Compare with H-UAV collaboration results to measure improvement."
    echo ""
fi
