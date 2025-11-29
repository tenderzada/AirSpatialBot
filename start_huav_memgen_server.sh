#!/bin/bash

# Start H-UAV server with MemGen-style Memory Weaver

echo "=========================================="
echo "Starting H-UAV MemGen Server"
echo "=========================================="

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
WEAVER_CHECKPOINT="./outputs/memgen_weaver_sqa/final"  # Set to trained checkpoint

# Server configuration
HOST="0.0.0.0"
PORT=8000

# Check if checkpoint exists
if [ -d "$WEAVER_CHECKPOINT" ]; then
    echo "Using trained MemGen Weaver from: $WEAVER_CHECKPOINT"
    CHECKPOINT_ARG="--weaver_checkpoint $WEAVER_CHECKPOINT"
else
    echo "WARNING: No checkpoint found at $WEAVER_CHECKPOINT"
    echo "Starting with untrained MemGen Weaver"
    CHECKPOINT_ARG=""
fi

# Start server
python hierarchical_uav/huav_memgen_server.py \
    --model_path $MODEL_PATH \
    $CHECKPOINT_ARG \
    --num_memory_tokens 8 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --host $HOST \
    --port $PORT

echo ""
echo "Server stopped."
