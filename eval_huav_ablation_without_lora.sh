#!/bin/bash

# H-UAV Ablation Study: WITHOUT LoRA (Baseline)
#
# 消融实验：测试无 LoRA 的基础 LLaVA 性能 (前100样本)

echo "============================================================"
echo "H-UAV Ablation Study: WITHOUT LoRA (Baseline)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TEST_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/ablation_huav_without_lora"
DEVICE="cuda:0"

# Ablation: Only first 100 samples
MAX_SAMPLES=100

echo ""
echo "🧪 Ablation Study Configuration:"
echo "  Condition: WITHOUT LoRA (Baseline LLaVA)"
echo "  Samples: First $MAX_SAMPLES samples"
echo "  Base model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""
echo "Expected: Lower performance without LoRA-enhanced memory"
echo "============================================================"

echo ""
echo "Starting H-UAV evaluation WITHOUT LoRA (Baseline)..."
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
    echo "✅ H-UAV WITHOUT LoRA (Baseline) Evaluation Complete"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/results.json"
    echo ""

    # Show quick stats if jq is available
    if command -v jq &> /dev/null; then
        if [ -f "$OUTPUT_DIR/results.json" ]; then
            echo "📊 Results (WITHOUT LoRA - Baseline):"
            echo "  MAE:  $(jq -r '.metrics.mae' $OUTPUT_DIR/results.json)"
            echo "  RMSE: $(jq -r '.metrics.rmse' $OUTPUT_DIR/results.json)"
            echo "  MRE:  $(jq -r '.metrics.mre' $OUTPUT_DIR/results.json)"
            echo ""
        fi
    fi

    echo "Comparison:"
    echo "  WITH LoRA:    ./outputs/ablation_huav_with_lora/results.json"
    echo "  WITHOUT LoRA: ./outputs/ablation_huav_without_lora/results.json"
    echo ""
else
    echo "❌ Evaluation failed (exit code: $EVAL_EXIT_CODE)"
fi
echo "============================================================"
