#!/bin/bash

# Train LoRA-Injected Memory Weaver for L-UAV (3 epochs)
#
# L-UAV 使用较弱的3 epoch LoRA，H-UAV使用更强的10 epoch LoRA

echo "============================================================"
echo "L-UAV Memory Weaver Training (LoRA Injection - 3 Epochs)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TRAIN_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/lora_injection_sqa_3epoch"
DEVICE="cuda:0"

# Training parameters
NUM_EPOCHS=3  # L-UAV uses weaker 3-epoch LoRA
BATCH_SIZE=4
LOG_INTERVAL=100
SAVE_INTERVAL=1

# Single GPU mode
GPUS="0"

# LoRA injection parameters (same as H-UAV)
LORA_RANK=8
LORA_ALPHA=16.0
TARGET_LAYERS="8 16 24"
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
echo "Architecture (L-UAV):"
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
echo "  - L-UAV with weaker 3-epoch LoRA (basic memory)"
echo "  - Can request enhanced memory from H-UAV (10-epoch LoRA)"
echo "  - Hierarchical UAV collaboration architecture"
echo "  - Save final checkpoint after 3 epochs"
echo ""
echo "Training:"
echo "  Epochs: $NUM_EPOCHS (weaker than H-UAV's 10 epochs)"
echo "  Batch size: $BATCH_SIZE"
echo "  GPUs: $GPUS"
echo "  Strategy: Save final checkpoint"
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

read -p "Start L-UAV training (3 epochs)? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "Starting L-UAV LoRA injection training (3 epochs)..."
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
    echo "L-UAV Training Complete (3 epochs) ✓"
else
    echo "L-UAV Training Failed ✗ (exit code: $TRAIN_EXIT_CODE)"
fi
echo "============================================================"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Outputs:"
    echo "  - LoRA adapters (final): $OUTPUT_DIR/lora_adapters_epochfinal.pt"
    echo "  - Training report: $OUTPUT_DIR/training_report.json"
    echo ""

    # Show size
    if [ -f "$OUTPUT_DIR/lora_adapters_epochfinal.pt" ]; then
        SIZE=$(du -h "$OUTPUT_DIR/lora_adapters_epochfinal.pt" | cut -f1)
        echo "✅ L-UAV LoRA adapters size: $SIZE"
    fi

    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. L-UAV uses this 3-epoch LoRA for basic memory capability"
    echo "  2. H-UAV uses 10-epoch LoRA for advanced memory capability"
    echo "  3. L-UAV requests memory enhancement from H-UAV when needed"
    echo ""
else
    echo ""
    echo "Training failed. Please check error messages."
    echo ""
fi
