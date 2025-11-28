#!/bin/bash

# Train LoRA-Injected Memory Weaver for H-UAV
#
# 记忆编织器：LoRA 适配器注入冻结的 LLaVA

echo "============================================================"
echo "H-UAV Memory Weaver Training (LoRA Injection)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TRAIN_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/lora_injection_sqa"
DEVICE="cuda:0"

# Training parameters
NUM_EPOCHS=10
BATCH_SIZE=4
LOG_INTERVAL=100
SAVE_INTERVAL=1  # Not used - only best model is saved

# Multi-GPU training (DataParallel)
# Note: DataParallel currently has compatibility issues with custom methods
# Using single GPU for stable training
GPUS="0"  # Single GPU mode

# LoRA injection parameters
LORA_RANK=8
LORA_ALPHA=16.0
TARGET_LAYERS="8 16 24"  # Inject at layers 8, 16, 24
TARGET_MODULES="q_proj v_proj"
NUM_MEMORY_TOKENS=8

# Optimizer
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Base model: $MODEL_PATH"
echo "  Training data: $TRAIN_DATA"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""
echo "Architecture:"
echo "  Frozen LLaVA (base)"
echo "    ↓"
echo "  LoRA injection at layers: $TARGET_LAYERS"
echo "  Target modules: $TARGET_MODULES"
echo "    ↓"
echo "  Generate memory tokens: $NUM_MEMORY_TOKENS"
echo ""
echo "LoRA Configuration:"
echo "  Rank: $LORA_RANK"
echo "  Alpha: $LORA_ALPHA"
echo "  Learning rate: $LEARNING_RATE"
echo ""
echo "Key Design:"
echo "  - Base LLaVA remains frozen"
echo "  - Only LoRA parameters are trainable"
echo "  - Memory generated from L-UAV hidden states"
echo "  - Returns latent memory sequence"
echo "  - Only best checkpoint is saved (based on loss)"
echo ""
echo "Training:"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  GPUs: $GPUS (DataParallel)"
echo "  Strategy: Save only the best model"
echo ""
echo "============================================================"
echo ""

# Check data
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Training data not found: $TRAIN_DATA"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ Image directory not found: $IMAGE_DIR"
    exit 1
fi

read -p "Start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "Starting LoRA injection training..."
echo ""

# Build command
CMD="python hierarchical_uav/train_lora_injection.py \
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
    --target_layers $TARGET_LAYERS \
    --num_memory_tokens $NUM_MEMORY_TOKENS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --gpus $GPUS \
    --load_8bit"

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
    echo "  - LoRA adapters (best): $OUTPUT_DIR/lora_adapters_best.pt"
    echo "  - Training report: $OUTPUT_DIR/training_report.json"
    echo ""

    # Show size
    if [ -f "$OUTPUT_DIR/lora_adapters_best.pt" ]; then
        SIZE=$(du -h "$OUTPUT_DIR/lora_adapters_best.pt" | cut -f1)
        echo "✅ Best LoRA adapters size: $SIZE"
    fi

    # Show best epoch info if report exists
    if [ -f "$OUTPUT_DIR/training_report.json" ] && command -v jq &> /dev/null; then
        BEST_EPOCH=$(jq -r '.best_model.epoch' $OUTPUT_DIR/training_report.json)
        BEST_LOSS=$(jq -r '.best_model.loss' $OUTPUT_DIR/training_report.json)
        echo "🏆 Best model: Epoch $BEST_EPOCH (Loss: $BEST_LOSS)"
    fi

    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Start H-UAV server with best LoRA checkpoint:"
    echo "     python hierarchical_uav/huav_lora_server_v2.py \\"
    echo "         --model_path $MODEL_PATH \\"
    echo "         --lora_weights $OUTPUT_DIR/lora_adapters_best.pt \\"
    echo "         --target_layers 8 16 24 \\"
    echo "         --port 8000"
    echo ""
    echo "  2. L-UAV sends hidden states to H-UAV:"
    echo "     POST http://h-uav:8000/get_memory"
    echo "     Body: {hidden_states: [...], shape: [1, 64, 4096]}"
    echo ""
    echo "  3. H-UAV returns memory tokens:"
    echo "     {memory_tokens: [[...], ...], memory_shape: [8, 4096]}"
    echo ""
else
    echo ""
    echo "Training failed. Please check error messages."
    echo ""
fi
