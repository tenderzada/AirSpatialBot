#!/bin/bash

# LoRA-Only Training Script for SQA Task
# Pure LoRA-based memory (NO MAC components)
# Designed for lightweight L-UAV deployment

echo "============================================================"
echo "LoRA-Only Memory Training for SQA Task"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TRAIN_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/lora_only_sqa"
DEVICE="cuda:0"

# Training parameters
NUM_EPOCHS=3
BATCH_SIZE=8
LOG_INTERVAL=100
SAVE_INTERVAL=1

# LoRA parameters (L-UAV optimized)
LORA_RANK=4              # Reduced for L-UAV (from 8)
LORA_ALPHA=8.0           # Scaled accordingly
NUM_MEMORY_TOKENS=8      # Moderate size (from 10)
LORA_LR=1e-4

# Optional features
ENABLE_TRIGGER=false     # Disable for L-UAV simplicity
POOL_METHOD="mean"       # mean, max, first, last

# Training options
USE_SUBSET=""            # Set to "--max-samples 1000" for testing

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Vision Tower: $VISION_TOWER"
echo "  Device: $DEVICE"
echo "  Training data: $TRAIN_DATA"
echo "  Image directory: $IMAGE_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Training parameters:"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LORA_LR"
echo ""
echo "LoRA configuration (L-UAV optimized):"
echo "  Rank: $LORA_RANK"
echo "  Alpha: $LORA_ALPHA"
echo "  Memory tokens: $NUM_MEMORY_TOKENS"
echo "  Trigger enabled: $ENABLE_TRIGGER"
echo "  Pool method: $POOL_METHOD"
echo ""
echo "Expected LoRA size:"
echo "  Parameters: ~180K"
echo "  File size: ~0.7 MB"
echo "  Compute overhead: ~0.5%"
echo ""
echo "SQA Dataset:"
echo "  Total samples: 17,526"
echo "  Question types: depth, distance, length, width, height, size"
echo ""

# Check data
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found at $TRAIN_DATA"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ Error: Image directory not found at $IMAGE_DIR"
    exit 1
fi

# Confirm
echo "⚠️  This will train LoRA-only memory (NO MAC components)"
echo "   Output will be deployable to L-UAV"
echo ""
read -p "Start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "Starting LoRA-only training..."
echo ""

# Build command
CMD="python hierarchical_uav/train_lora_only.py \
    --model_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --device $DEVICE \
    --train_data $TRAIN_DATA \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --num_memory_tokens $NUM_MEMORY_TOKENS \
    --lora_lr $LORA_LR \
    --pool_method $POOL_METHOD \
    --load_8bit \
    $USE_SUBSET"

if [ "$ENABLE_TRIGGER" = true ]; then
    CMD="$CMD --enable_trigger"
fi

# Run training
eval $CMD

TRAIN_EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training Complete ✓"
else
    echo "Training Failed ✗ (exit code: $TRAIN_EXIT_CODE)"
fi
echo "============================================================"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Outputs:"
    echo "  - LoRA weights: $OUTPUT_DIR/lora_weights_final.pt"
    echo "  - Training report: $OUTPUT_DIR/training_report.json"
    echo ""

    # Show file size
    if [ -f "$OUTPUT_DIR/lora_weights_final.pt" ]; then
        SIZE=$(du -h "$OUTPUT_DIR/lora_weights_final.pt" | cut -f1)
        echo "✅ LoRA weights size: $SIZE (should be <1 MB)"
    fi

    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Deploy to L-UAV:"
    echo "     scp $OUTPUT_DIR/lora_weights_final.pt luav:/path/to/weights/"
    echo ""
    echo "  2. Evaluate on L-UAV:"
    echo "     python hierarchical_uav/eval_sqa.py \\"
    echo "         --uav_type l-uav \\"
    echo "         --lora_only \\"
    echo "         --load-lora $OUTPUT_DIR/lora_weights_final.pt \\"
    echo "         --device cuda:0"
    echo ""
    echo "  3. Compare with baseline:"
    echo "     ./run_sqa_evaluation.sh l-uav --standalone"
    echo ""
else
    echo ""
    echo "Training failed. Please check:"
    echo "  1. Data paths are correct"
    echo "  2. GPU memory available"
    echo "  3. Error messages above"
    echo ""
fi
