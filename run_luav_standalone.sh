#!/bin/bash

# L-UAV Standalone Inference Script
# Baseline performance without H-UAV memory augmentation

echo "============================================================"
echo "Starting L-UAV (Standalone Mode - No H-UAV Memory)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TEST_DATA="/mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/hierarchical_uav"

# Standalone Mode Configuration
THRESHOLD=0.0  # Set to 0.0 to disable H-UAV queries (always use local inference)
FORCE_QUERY_RATE=0.0  # Set to 0.0 to use KB when available
HUAV_ADDRESS="localhost:50051"  # Dummy address (won't be used)

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Vision Tower: $VISION_TOWER"
echo "  Device: cuda:1"
echo "  Mode: Standalone (No H-UAV)"
echo "  Self-matching Threshold: $THRESHOLD (disabled)"
echo "  Force Query Rate: $FORCE_QUERY_RATE (KB enabled)"
echo "  Output: $OUTPUT_DIR/task1_l-uav_standalone.jsonl"
echo ""
echo "This will establish baseline L-UAV performance without memory augmentation."
echo ""

# Run L-UAV in standalone mode
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device cuda:1 \
    --standalone \
    --threshold $THRESHOLD \
    --force-query-rate $FORCE_QUERY_RATE \
    --model_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --test_data $TEST_DATA \
    --image_dir $IMAGE_DIR \
    --load_8bit \
    --output $OUTPUT_DIR/task1_l-uav_standalone.jsonl 2>&1 | tee $OUTPUT_DIR/luav_standalone.log

echo ""
echo "============================================================"
echo "L-UAV Standalone Inference Completed"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR/task1_l-uav_standalone.jsonl"
echo "Log saved to: $OUTPUT_DIR/luav_standalone.log"
echo ""
echo "To analyze results:"
echo "  python analyze_uav_results.py $OUTPUT_DIR/task1_l-uav_standalone.jsonl"
echo ""
echo "Statistics:"
jq -r '.source' $OUTPUT_DIR/task1_l-uav_standalone.jsonl 2>/dev/null | sort | uniq -c | awk '{print "  " $2 ": " $1}'
echo ""
total=$(jq -s 'length' $OUTPUT_DIR/task1_l-uav_standalone.jsonl 2>/dev/null)
correct=$(jq -s '[.[] | select(.answer == .ground_truth or (.ground_truth | tostring | ascii_downcase | contains(.answer | tostring | ascii_downcase)))] | length' $OUTPUT_DIR/task1_l-uav_standalone.jsonl 2>/dev/null)
if [ "$total" -gt 0 ]; then
    accuracy=$(echo "scale=4; $correct * 100 / $total" | bc)
    echo "Baseline Accuracy: $correct / $total = $accuracy%"
fi
echo ""
