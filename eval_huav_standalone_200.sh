#!/bin/bash

# H-UAV Standalone Evaluation (200 samples for comparison)
#
# 使用训练好的 LoRA Memory Weaver，评估前 200 个样本

echo "============================================================"
echo "Comparison Experiment 1: H-UAV Standalone with MemGen LoRA"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
LORA_WEIGHTS="./outputs/lora_injection_sqa/lora_adapters_epochbest.pt"
TEST_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/comparison_huav_standalone_200"
DEVICE="cuda:0"

# LoRA configuration
LORA_RANK=8
TARGET_LAYERS="8 16 24"

# Comparison: First 200 samples
MAX_SAMPLES=200

echo ""
echo "🧪 Comparison Experiment Configuration:"
echo "  Condition: H-UAV Standalone with LoRA MemGen"
echo "  Samples: First $MAX_SAMPLES samples"
echo "  LoRA weights: $LORA_WEIGHTS (best from 10 epochs)"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""
echo "Expected: High performance with LoRA-enhanced memory"
echo "============================================================"

# Check LoRA weights
if [ ! -f "$LORA_WEIGHTS" ]; then
    echo ""
    echo "❌ LoRA weights not found: $LORA_WEIGHTS"
    echo ""
    echo "Please train the model first:"
    echo "  ./train_lora_injection_sqa.sh"
    echo ""
    exit 1
fi

echo ""
echo "Starting H-UAV standalone evaluation (with MemGen LoRA)..."
echo ""

# Run evaluation
python hierarchical_uav/eval_huav_standalone.py \
    --model_path "$MODEL_PATH" \
    --vision_tower "$VISION_TOWER" \
    --lora_weights "$LORA_WEIGHTS" \
    --device "$DEVICE" \
    --lora_rank $LORA_RANK \
    --target_layers $TARGET_LAYERS \
    --test_data "$TEST_DATA" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples $MAX_SAMPLES \
    --load_8bit

EVAL_EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ H-UAV Standalone Evaluation Complete"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/results.json"
    echo ""

    # Show quick stats if jq is available
    if command -v jq &> /dev/null; then
        if [ -f "$OUTPUT_DIR/results.json" ]; then
            echo "📊 Results (H-UAV Standalone with MemGen):"
            echo "  MAE:  $(jq -r '.metrics.mae' $OUTPUT_DIR/results.json)"
            echo "  RMSE: $(jq -r '.metrics.rmse' $OUTPUT_DIR/results.json)"
            echo "  MRE:  $(jq -r '.metrics.mre' $OUTPUT_DIR/results.json)"
            echo ""
        fi
    fi

    echo "Next: Run L-UAV + H-UAV collaboration experiment"
    echo "  ./eval_luav_huav_collab_200.sh"
    echo ""
else
    echo "❌ Evaluation failed (exit code: $EVAL_EXIT_CODE)"
fi
echo "============================================================"
