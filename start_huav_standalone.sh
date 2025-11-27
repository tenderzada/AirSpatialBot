#!/bin/bash

# H-UAV Standalone Evaluation
#
# 直接使用 LoRA 记忆编织器进行推理，测试 MemGen 效果

echo "============================================================"
echo "H-UAV Standalone SQA Evaluation (LoRA Memory Weaver)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
LORA_WEIGHTS="./outputs/lora_injection_sqa/lora_adapters_epochfinal.pt"
TEST_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/eval_huav_standalone"
DEVICE="cuda:0"

# LoRA configuration
LORA_RANK=8
TARGET_LAYERS="8 16 24"

# Optional: limit samples for quick testing
MAX_SAMPLES=""  # Leave empty for all samples

echo ""
echo "Configuration:"
echo "  Base model: $MODEL_PATH"
echo "  Vision tower: $VISION_TOWER"
echo "  LoRA weights: $LORA_WEIGHTS"
echo "  Test data: $TEST_DATA"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""
echo "LoRA Configuration:"
echo "  Rank: $LORA_RANK"
echo "  Target layers: $TARGET_LAYERS"
echo ""
echo "Architecture:"
echo "  ┌─────────────────────────────────┐"
echo "  │  Image + Question               │"
echo "  └─────────────────────────────────┘"
echo "           ↓"
echo "  ┌─────────────────────────────────┐"
echo "  │  Frozen Base LLaVA (7B)         │"
echo "  └─────────────────────────────────┘"
echo "           ↓"
echo "  ┌─────────────────────────────────┐"
echo "  │  LoRA @ Layers 8, 16, 24        │"
echo "  │  (q_proj, v_proj)               │"
echo "  │  Trainable: 8.4M params         │"
echo "  └─────────────────────────────────┘"
echo "           ↓"
echo "  ┌─────────────────────────────────┐"
echo "  │  Memory-Enhanced Hidden States  │"
echo "  └─────────────────────────────────┘"
echo "           ↓"
echo "  ┌─────────────────────────────────┐"
echo "  │  Final Answer                   │"
echo "  └─────────────────────────────────┘"
echo ""
echo "Purpose:"
echo "  - Test LoRA memory weaver effectiveness"
echo "  - Measure MemGen performance on SQA"
echo "  - Compare with baseline (L-UAV standalone)"
echo ""
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
echo "LoRA weights found ✓"
WEIGHTS_SIZE=$(du -h "$LORA_WEIGHTS" | cut -f1)
echo "  Size: $WEIGHTS_SIZE"
echo ""

echo "Starting H-UAV standalone evaluation..."
echo ""

# Build command
CMD="python hierarchical_uav/eval_huav_standalone.py \
    --model_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --lora_weights $LORA_WEIGHTS \
    --device $DEVICE \
    --lora_rank $LORA_RANK \
    --target_layers $TARGET_LAYERS \
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
    echo "H-UAV Standalone Evaluation Complete ✓"
else
    echo "H-UAV Standalone Evaluation Failed ✗ (exit code: $EVAL_EXIT_CODE)"
fi
echo "============================================================"

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Results saved to: $OUTPUT_DIR/results.json"
    echo ""

    # Show quick stats if jq is available
    if command -v jq &> /dev/null; then
        if [ -f "$OUTPUT_DIR/results.json" ]; then
            echo "Overall Metrics:"
            echo "  MAE:  $(jq -r '.metrics.mae' $OUTPUT_DIR/results.json)"
            echo "  RMSE: $(jq -r '.metrics.rmse' $OUTPUT_DIR/results.json)"
            echo "  MRE:  $(jq -r '.metrics.mre' $OUTPUT_DIR/results.json)"
            echo ""

            # Show per-type metrics if available
            if jq -e '.type_metrics' $OUTPUT_DIR/results.json > /dev/null 2>&1; then
                echo "Per-Type Metrics:"
                jq -r '.type_metrics | to_entries[] | "  \(.key): MAE=\(.value.mae | tostring), RMSE=\(.value.rmse | tostring)"' $OUTPUT_DIR/results.json
                echo ""
            fi
        fi
    fi

    echo "This tests the LoRA memory weaver directly."
    echo ""
    echo "Compare with:"
    echo "  - L-UAV standalone (baseline): ./outputs/eval_luav_standalone/results.json"
    echo "  - L-UAV with H-UAV (collaborative): ./outputs/eval_luav_with_huav/results.json"
    echo ""
    echo "To view full results:"
    echo "  cat $OUTPUT_DIR/results.json | jq ."
    echo ""
fi
