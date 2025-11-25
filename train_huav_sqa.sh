#!/bin/bash

# H-UAV Test-Time Learning Script for SQA Task
# Train MAC memory on Spatial Question Answering dataset

echo "============================================================"
echo "H-UAV Test-Time Learning for SQA Task"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TRAIN_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/huav_training_sqa"
DEVICE="cuda:0"

# Training parameters
NUM_EPOCHS=3
LOG_INTERVAL=100
SAVE_INTERVAL=1

# MAC parameters (same as Task1 for consistency)
LEARNING_THETA=0.1    # Learning rate for memory update
SURPRISE_ETA=0.9      # Surprise decay (momentum)
FORGETTING_ALPHA=0.01 # Forgetting rate (weight decay)

# SQA-specific options
USE_SUBSET=""         # Set to "--max-samples 1000" to train on subset for testing

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
echo "  Learning rate (theta): $LEARNING_THETA"
echo "  Surprise decay (eta): $SURPRISE_ETA"
echo "  Forgetting rate (alpha): $FORGETTING_ALPHA"
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
echo "Starting H-UAV training on SQA task..."
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
    --load_8bit \
    $USE_SUBSET

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
    echo "Next steps:"
    echo ""
    echo "  1. Review training report:"
    echo "     cat $OUTPUT_DIR/training_report.json | jq '.'"
    echo ""
    echo "  2. Start H-UAV server with trained memory:"
    echo "     python hierarchical_uav/eval_sqa.py \\"
    echo "         --uav_type h-uav \\"
    echo "         --device cuda:0 \\"
    echo "         --port 50051 \\"
    echo "         --load-memory $OUTPUT_DIR/huav_memory_final.pt"
    echo ""
    echo "  3. Run L-UAV evaluation (in another terminal):"
    echo "     python hierarchical_uav/eval_sqa.py \\"
    echo "         --uav_type l-uav \\"
    echo "         --device cuda:1 \\"
    echo "         --huav_address localhost:50051"
    echo ""
    echo "  4. Or use the convenient runner:"
    echo "     # Terminal 1: Start H-UAV with trained memory"
    echo "     H-UAV_MEMORY=$OUTPUT_DIR/huav_memory_final.pt ./run_sqa_evaluation.sh h-uav"
    echo ""
    echo "     # Terminal 2: Start L-UAV"
    echo "     ./run_sqa_evaluation.sh l-uav"
    echo ""
else
    echo ""
    echo "Please check the error messages above and:"
    echo "  1. Verify data paths are correct"
    echo "  2. Ensure images are accessible"
    echo "  3. Check GPU memory availability (use nvidia-smi)"
    echo ""
fi
