#!/bin/bash

# Start H-UAV LoRA Memory Server
#
# H-UAV serves as the central memory provider for L-UAVs
# using LoRA-based memory synthesis (MemGen-inspired)

echo "============================================================"
echo "H-UAV LoRA Memory Server"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
LORA_MEMORY="./outputs/lora_only_sqa/lora_weights_final.pt"
HOST="0.0.0.0"
PORT=8000
DEVICE="cuda:0"

# LoRA configuration
LORA_RANK=4
NUM_MEMORY_TOKENS=8

echo ""
echo "Server Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Device: $DEVICE"
echo ""
echo "Model Configuration:"
echo "  Base model: $MODEL_PATH"
echo "  Vision tower: $VISION_TOWER"
echo "  LoRA memory: $LORA_MEMORY"
echo ""
echo "LoRA Memory Configuration:"
echo "  LoRA rank: $LORA_RANK"
echo "  Memory tokens: $NUM_MEMORY_TOKENS"
echo "  Expected size: ~0.7 MB"
echo ""

# Check if LoRA weights exist
if [ ! -f "$LORA_MEMORY" ]; then
    echo "⚠️  LoRA weights not found at: $LORA_MEMORY"
    echo ""
    echo "Please train LoRA memory first:"
    echo "  ./train_lora_only_sqa.sh"
    echo ""
    read -p "Start server without pre-trained LoRA? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Server startup cancelled."
        exit 1
    fi
    LORA_MEMORY=""
fi

echo "============================================================"
echo "Architecture:"
echo ""
echo "  L-UAV → Inference → Self-match (low confidence?)"
echo "                              ↓"
echo "  L-UAV → Request H-UAV for memory tokens"
echo "                              ↓"
echo "  H-UAV (this server) → Generate LoRA Memory Tokens"
echo "                              ↓"
echo "  Return memory tokens [8, 4096] → L-UAV"
echo "                              ↓"
echo "  L-UAV → Enhanced inference with memory → Answer"
echo ""
echo "============================================================"
echo "KEY: H-UAV returns MEMORY TOKENS, NOT answers!"
echo "     L-UAV uses memory to enhance its own inference"
echo "============================================================"
echo ""
echo "Endpoints (when started):"
echo "  Health check: http://$HOST:$PORT/health"
echo "  Get memory: http://$HOST:$PORT/get_memory"
echo "  Statistics: http://$HOST:$PORT/stats"
echo ""
echo "============================================================"
echo ""
read -p "Start H-UAV server? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Server startup cancelled."
    exit 0
fi

echo ""
echo "Starting H-UAV LoRA Memory Server..."
echo ""

# Build command
CMD="python hierarchical_uav/huav_lora_server.py \
    --model_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --host $HOST \
    --port $PORT \
    --device $DEVICE \
    --lora_rank $LORA_RANK \
    --num_memory_tokens $NUM_MEMORY_TOKENS"

if [ -n "$LORA_MEMORY" ]; then
    CMD="$CMD --lora_memory $LORA_MEMORY"
fi

# Run server
eval $CMD

SERVER_EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $SERVER_EXIT_CODE -eq 0 ]; then
    echo "Server stopped normally ✓"
else
    echo "Server stopped with error ✗ (exit code: $SERVER_EXIT_CODE)"
fi
echo "============================================================"
