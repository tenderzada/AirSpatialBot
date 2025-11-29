#!/bin/bash

# H-UAV Baseline Evaluation (WITHOUT LoRA, 200 samples for comparison)
#
# 评估无 LoRA 的基础 LLaVA，作为 baseline

echo "============================================================"
echo "Comparison Experiment 3: H-UAV Baseline (WITHOUT MemGen)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TEST_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/comparison_huav_baseline_200"
DEVICE="cuda:0"

# Comparison: First 200 samples
MAX_SAMPLES=200

echo ""
echo "🧪 Comparison Experiment Configuration:"
echo "  Condition: H-UAV Baseline WITHOUT LoRA MemGen"
echo "  Samples: First $MAX_SAMPLES samples"
echo "  Model: Vanilla LLaVA (no memory enhancement)"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""
echo "Expected: Baseline performance without LoRA-enhanced memory"
echo "============================================================"

echo ""
echo "Starting H-UAV baseline evaluation (WITHOUT MemGen)..."
echo ""

# Run evaluation
python hierarchical_uav/eval_huav_baseline.py \
    --model_path "$MODEL_PATH" \
    --vision_tower "$VISION_TOWER" \
    --device "$DEVICE" \
    --test_data "$TEST_DATA" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples $MAX_SAMPLES \
    --load_8bit

EVAL_EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ H-UAV Baseline Evaluation Complete"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/results.json"
    echo ""

    # Show quick stats if jq is available
    if command -v jq &> /dev/null; then
        if [ -f "$OUTPUT_DIR/results.json" ]; then
            echo "📊 Results (H-UAV Baseline WITHOUT MemGen):"
            echo "  MAE:  $(jq -r '.metrics.mae' $OUTPUT_DIR/results.json)"
            echo "  RMSE: $(jq -r '.metrics.rmse' $OUTPUT_DIR/results.json)"
            echo "  MRE:  $(jq -r '.metrics.mre' $OUTPUT_DIR/results.json)"
            echo ""
        fi
    fi

    echo "This establishes the baseline for MemGen comparison"
    echo ""
else
    echo "❌ Evaluation failed (exit code: $EVAL_EXIT_CODE)"
fi
echo "============================================================"
