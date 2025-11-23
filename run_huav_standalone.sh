#!/bin/bash

# H-UAV Standalone Inference Script
# Evaluate H-UAV with trained MAC memory (no server mode)

echo "============================================================"
echo "Starting H-UAV (Standalone Mode - Direct Inference)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TEST_DATA="/mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/hierarchical_uav"

# MAC Memory Configuration
MEMORY_CHECKPOINT="${LOAD_MEMORY:-./outputs/huav_training/huav_memory_final.pt}"

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Vision Tower: $VISION_TOWER"
echo "  Device: cuda:0"
echo "  Mode: Standalone (Direct Inference)"
echo "  MAC Memory: $MEMORY_CHECKPOINT"
echo "  Output: $OUTPUT_DIR/task1_h-uav_standalone.jsonl"
echo ""

# Check if memory checkpoint exists
if [ -f "$MEMORY_CHECKPOINT" ]; then
    echo "✓ Found trained MAC memory checkpoint"
    LOAD_MEMORY_FLAG="--load-memory $MEMORY_CHECKPOINT"
else
    echo "⚠ Warning: MAC memory checkpoint not found at $MEMORY_CHECKPOINT"
    echo "  Will use random initialization"
    LOAD_MEMORY_FLAG=""
fi

echo ""
echo "This will evaluate H-UAV performance with trained MAC memory."
echo ""

# Run H-UAV in standalone evaluation mode
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cuda:0 \
    --standalone \
    $LOAD_MEMORY_FLAG \
    --model_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --test_data $TEST_DATA \
    --image_dir $IMAGE_DIR \
    --load_8bit \
    --output $OUTPUT_DIR/task1_h-uav_standalone.jsonl 2>&1 | tee $OUTPUT_DIR/huav_standalone.log

echo ""
echo "============================================================"
echo "H-UAV Standalone Inference Completed"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR/task1_h-uav_standalone.jsonl"
echo "Log saved to: $OUTPUT_DIR/huav_standalone.log"
echo ""
echo "To analyze results:"
echo "  python analyze_uav_results.py $OUTPUT_DIR/task1_h-uav_standalone.jsonl"
echo ""
echo "Statistics:"
jq -r '.source' $OUTPUT_DIR/task1_h-uav_standalone.jsonl 2>/dev/null | sort | uniq -c | awk '{print "  " $2 ": " $1}'
echo ""
total=$(jq -s 'length' $OUTPUT_DIR/task1_h-uav_standalone.jsonl 2>/dev/null)
correct=$(jq -s '[.[] | select(.answer == .ground_truth or (.ground_truth | tostring | ascii_downcase | contains(.answer | tostring | ascii_downcase)))] | length' $OUTPUT_DIR/task1_h-uav_standalone.jsonl 2>/dev/null)
if [ "$total" -gt 0 ]; then
    accuracy=$(echo "scale=4; $correct * 100 / $total" | bc)
    echo "H-UAV Standalone Accuracy: $correct / $total = $accuracy%"
fi
echo ""
echo "To compare L-UAV vs H-UAV standalone:"
echo "  python compare_baseline_vs_memory.py \\"
echo "    $OUTPUT_DIR/task1_l-uav_standalone.jsonl \\"
echo "    $OUTPUT_DIR/task1_h-uav_standalone.jsonl"
echo ""
