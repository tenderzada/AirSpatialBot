#!/bin/bash

# Training script for MemGen-style Memory Weaver
# This trains the learnable query latents and LoRA adapters

echo "=========================================="
echo "Training MemGen-style Memory Weaver"
echo "=========================================="

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
TRAIN_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
OUTPUT_DIR="./outputs/memgen_weaver_sqa"

# MemGen parameters (following official implementation)
NUM_MEMORY_TOKENS=8
LORA_RANK=16
LORA_ALPHA=32.0
LORA_DROPOUT=0.1

# Training parameters
NUM_EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=1e-5
WEIGHT_DECAY=0.01

# Run training
python hierarchical_uav/train_memgen_weaver.py \
    --model_path $MODEL_PATH \
    --train_data $TRAIN_DATA \
    --output_dir $OUTPUT_DIR \
    --num_memory_tokens $NUM_MEMORY_TOKENS \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --log_interval 10 \
    --save_interval 1

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Checkpoint saved to: $OUTPUT_DIR"
echo "=========================================="
