#!/bin/bash

# H-UAV Test-Time Learning Script
# Phase 3: Train MAC memory to learn meaningful representations

echo "============================================================"
echo "H-UAV Test-Time Learning (Phase 3)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TRAIN_DATA="/mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/huav_training"
DEVICE="cuda:0"

# Training parameters
NUM_EPOCHS=3
LOG_INTERVAL=100
SAVE_INTERVAL=1

# MAC parameters
LEARNING_THETA=0.1    # Learning rate for memory update
SURPRISE_ETA=0.9      # Surprise decay (momentum)
FORGETTING_ALPHA=0.01 # Forgetting rate (weight decay)

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Vision Tower: $VISION_TOWER"
echo "  Device: $DEVICE"
echo "  Training data: $TRAIN_DATA"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Training parameters:"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate (theta): $LEARNING_THETA"
echo "  Surprise decay (eta): $SURPRISE_ETA"
echo "  Forgetting rate (alpha): $FORGETTING_ALPHA"
echo ""

# Confirm before starting
read -p "Start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "Starting H-UAV training..."
echo ""

# Run training
python hierarchical_uav/train_huav.py \
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
    --load_8bit

echo ""
echo "============================================================"
echo "Training Complete"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review training report:"
echo "     cat $OUTPUT_DIR/training_report.json | jq '.'"
echo ""
echo "  2. Start H-UAV server with trained memory:"
echo "     LOAD_MEMORY=$OUTPUT_DIR/huav_memory_final.pt ./run_huav.sh"
echo ""
echo "  3. Run L-UAV evaluation to test memory effectiveness"
echo ""
