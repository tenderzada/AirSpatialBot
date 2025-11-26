#!/bin/bash

# Hybrid MAC-LoRA Training Script for SQA Task
# Trains both MAC memory and LoRA adapters

echo "============================================================"
echo "Hybrid MAC-LoRA Training for SQA Task"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TRAIN_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/hybrid_lora_training_sqa"
DEVICE="cuda:0"

# Training parameters
NUM_EPOCHS=1
LOG_INTERVAL=100
SAVE_INTERVAL=1

# MAC parameters
LEARNING_THETA=0.1
SURPRISE_ETA=0.9
FORGETTING_ALPHA=0.01

# LoRA parameters
LORA_RANK=8
LORA_ALPHA=16.0
LORA_LR=1e-4

# Hybrid parameters
FUSION_MODE="concat"  # concat, add, learned, adaptive
ENABLE_TRIGGER=true
TRIGGER_THRESHOLD=0.5

# Training mode
TRAINING_MODE="hybrid"  # Options: hybrid, lora_only, mac_only

# SQA-specific options
USE_SUBSET=""  # Set to "--max-samples 1000" for testing

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
echo "  Mode: $TRAINING_MODE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Fusion mode: $FUSION_MODE"
echo ""
echo "MAC parameters:"
echo "  Learning rate (theta): $LEARNING_THETA"
echo "  Surprise decay (eta): $SURPRISE_ETA"
echo "  Forgetting rate (alpha): $FORGETTING_ALPHA"
echo ""
echo "LoRA parameters:"
echo "  Rank: $LORA_RANK"
echo "  Alpha: $LORA_ALPHA"
echo "  Learning rate: $LORA_LR"
echo ""
echo "Adaptive trigger:"
echo "  Enabled: $ENABLE_TRIGGER"
echo "  Threshold: $TRIGGER_THRESHOLD"
echo ""
echo "SQA Dataset Info:"
echo "  Total samples: 17,526 (6 question types)"
echo "  Question types: depth, distance, length, width, height, size"
echo "  Task: Spatial regression (numeric predictions)"
echo ""

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found at $TRAIN_DATA"
    echo ""
    echo "Please ensure the SQA dataset is downloaded:"
    echo "  - Expected path: $TRAIN_DATA"
    echo "  - Dataset: airspatial_sqa_test.jsonl (17,526 samples)"
    echo ""
    exit 1
fi

# Check if image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ Error: Image directory not found at $IMAGE_DIR"
    echo ""
    echo "Please ensure images are available:"
    echo "  - Expected path: $IMAGE_DIR"
    echo ""
    exit 1
fi

# Confirm before starting
echo "⚠️  Training will process 17,526 samples × $NUM_EPOCHS epochs"
echo "   Estimated time: 2-4 hours on GPU"
echo ""
read -p "Start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "Starting Hybrid MAC-LoRA training on SQA task..."
echo ""

# Build command
CMD="python hierarchical_uav/train_hybrid_lora.py \
    --model_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --device $DEVICE \
    --train_data $TRAIN_DATA \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --learning_theta $LEARNING_THETA \
    --surprise_eta $SURPRISE_ETA \
    --forgetting_alpha $FORGETTING_ALPHA \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_lr $LORA_LR \
    --fusion_mode $FUSION_MODE \
    --trigger_threshold $TRIGGER_THRESHOLD \
    --training_mode $TRAINING_MODE \
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
    echo "  - Hybrid memory: $OUTPUT_DIR/hybrid_memory_final.pt"
    echo "  - LoRA weights: $OUTPUT_DIR/lora_weights_final.pt"
    echo "  - Training report: $OUTPUT_DIR/training_report.json"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Review training report:"
    echo "     cat $OUTPUT_DIR/training_report.json | jq '.'"
    echo ""
    echo "  2. Run H-UAV evaluation with hybrid memory:"
    echo "     python hierarchical_uav/eval_sqa.py \\"
    echo "         --uav_type h-uav \\"
    echo "         --device cuda:0 \\"
    echo "         --hybrid_lora \\"
    echo "         --load-memory $OUTPUT_DIR/hybrid_memory_final.pt \\"
    echo "         --standalone"
    echo ""
    echo "  3. Compare with baseline MAC:"
    echo "     ./run_sqa_evaluation.sh h-uav --standalone"
    echo ""
else
    echo ""
    echo "Please check the error messages above and:"
    echo "  1. Verify data paths are correct"
    echo "  2. Ensure images are accessible"
    echo "  3. Check GPU memory availability (use nvidia-smi)"
    echo ""
fi
