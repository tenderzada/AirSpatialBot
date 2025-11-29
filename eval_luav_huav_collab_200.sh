#!/bin/bash

# L-UAV + H-UAV Collaboration Evaluation (200 samples for comparison)
#
# L-UAV 每隔 5 个样本向 H-UAV 发起 query，评估协作效果

echo "============================================================"
echo "Comparison Experiment 2: L-UAV + H-UAV Collaboration"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TEST_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/comparison_luav_huav_collab_200"

# L-UAV device
LUAV_DEVICE="cuda:1"

# H-UAV connection
HUAV_ADDRESS="localhost:50051"
CONFIDENCE_THRESHOLD=0.5

# Force H-UAV query every N samples
FORCE_HUAV_EVERY_N=5

# Comparison: First 200 samples
MAX_SAMPLES=200

echo ""
echo "🧪 Comparison Experiment Configuration:"
echo "  Condition: L-UAV + H-UAV Collaboration"
echo "  Samples: First $MAX_SAMPLES samples"
echo "  L-UAV device: $LUAV_DEVICE"
echo "  H-UAV address: $HUAV_ADDRESS"
echo "  Force query: Every $FORCE_HUAV_EVERY_N samples"
echo "  Confidence threshold: $CONFIDENCE_THRESHOLD"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Collaboration Strategy:"
echo "  - L-UAV performs base inference"
echo "  - Force H-UAV query every $FORCE_HUAV_EVERY_N samples"
echo "  - Also query if confidence < $CONFIDENCE_THRESHOLD"
echo "  - H-UAV provides LoRA-enhanced memory tokens"
echo ""
echo "Expected: Performance boost from H-UAV memory guidance"
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
echo "Starting L-UAV + H-UAV collaboration evaluation..."
echo ""

# Run evaluation
python hierarchical_uav/eval_sqa_lora_socket.py \
    --model_path "$MODEL_PATH" \
    --vision_tower "$VISION_TOWER" \
    --device "$LUAV_DEVICE" \
    --huav_address "$HUAV_ADDRESS" \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --force_huav_every_n $FORCE_HUAV_EVERY_N \
    --test_data "$TEST_DATA" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples $MAX_SAMPLES \
    --load_8bit

EVAL_EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ L-UAV + H-UAV Collaboration Evaluation Complete"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/results.json"
    echo ""

    # Show quick stats if jq is available
    if command -v jq &> /dev/null; then
        if [ -f "$OUTPUT_DIR/results.json" ]; then
            echo "📊 Results (L-UAV + H-UAV Collaboration):"
            echo "  MAE:  $(jq -r '.metrics.mae' $OUTPUT_DIR/results.json)"
            echo "  RMSE: $(jq -r '.metrics.rmse' $OUTPUT_DIR/results.json)"
            echo "  MRE:  $(jq -r '.metrics.mre' $OUTPUT_DIR/results.json)"
            echo ""
            echo "Collaboration Stats:"
            echo "  H-UAV usage rate: $(jq -r '.collaboration_stats.huav_usage_rate' $OUTPUT_DIR/results.json)"
            echo ""
        fi
    fi

    echo "Next: Compare all results"
    echo "  python compare_collaboration_results.py"
    echo ""
else
    echo "❌ Evaluation failed (exit code: $EVAL_EXIT_CODE)"
fi
echo "============================================================"
