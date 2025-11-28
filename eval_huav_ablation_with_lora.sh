#!/bin/bash

# H-UAV Ablation Study: WITH LoRA Memory Weaver
#
# 消融实验：测试有 LoRA 的 H-UAV 性能 (前100样本)

echo "============================================================"
echo "H-UAV Ablation Study: WITH LoRA Memory Weaver"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
LORA_WEIGHTS="./outputs/lora_injection_sqa/lora_adapters_epochfinal.pt"
TEST_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/ablation_huav_with_lora"
DEVICE="cuda:0"

# LoRA configuration
LORA_RANK=8
TARGET_LAYERS="8 16 24"

# Ablation: Only first 100 samples
MAX_SAMPLES=100

echo ""
echo "🧪 Ablation Study Configuration:"
echo "  Condition: WITH LoRA Memory Weaver"
echo "  Samples: First $MAX_SAMPLES samples"
echo "  Base model: $MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""
echo "LoRA Configuration:"
echo "  Rank: $LORA_RANK"
echo "  Target layers: $TARGET_LAYERS"
echo ""
echo "Expected: Higher performance with LoRA-enhanced memory"
echo "============================================================"

# Check LoRA weights
if [ ! -f "$LORA_WEIGHTS" ]; then
    echo ""
    echo "❌ LoRA weights not found: $LORA_WEIGHTS"
    echo ""
    exit 1
fi

echo ""
echo "Starting H-UAV evaluation WITH LoRA..."
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
    echo "✅ H-UAV WITH LoRA Evaluation Complete"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/results.json"
    echo ""

    # Show quick stats if jq is available
    if command -v jq &> /dev/null; then
        if [ -f "$OUTPUT_DIR/results.json" ]; then
            echo "📊 Results (WITH LoRA):"
            echo "  MAE:  $(jq -r '.metrics.mae' $OUTPUT_DIR/results.json)"
            echo "  RMSE: $(jq -r '.metrics.rmse' $OUTPUT_DIR/results.json)"
            echo "  MRE:  $(jq -r '.metrics.mre' $OUTPUT_DIR/results.json)"
            echo ""
        fi
    fi

    echo "Next step:"
    echo "  Run WITHOUT LoRA for comparison:"
    echo "  ./eval_huav_ablation_without_lora.sh"
    echo ""
else
    echo "❌ Evaluation failed (exit code: $EVAL_EXIT_CODE)"
fi
echo "============================================================"
