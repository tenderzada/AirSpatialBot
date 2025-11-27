#!/bin/bash

# Start H-UAV LoRA Memory Server (Socket-based)
#
# 使用Socket通信（与之前gRPC风格一致）

echo "============================================================"
echo "H-UAV LoRA Memory Server (Socket-based)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
LORA_WEIGHTS="./outputs/lora_injection_sqa/lora_adapters_epochfinal.pt"
PORT=50051
DEVICE="cuda:0"

# LoRA injection configuration
TARGET_LAYERS="8 16 24"
LORA_RANK=8

echo ""
echo "Configuration:"
echo "  Base model: $MODEL_PATH"
echo "  Vision tower: $VISION_TOWER"
echo "  LoRA weights: $LORA_WEIGHTS"
echo "  Target layers: $TARGET_LAYERS"
echo "  LoRA rank: $LORA_RANK"
echo "  Port: $PORT (Socket)"
echo "  Device: $DEVICE"
echo ""
echo "Communication: Socket-based (pickle protocol)"
echo "  - Lightweight, low latency"
echo "  - Compatible with previous gRPC-style implementation"
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

echo "Starting H-UAV LoRA server (Socket)..."
echo ""

python hierarchical_uav/huav_lora_server_socket.py \
    --model_path "$MODEL_PATH" \
    --vision_tower "$VISION_TOWER" \
    --lora_weights "$LORA_WEIGHTS" \
    --target_layers $TARGET_LAYERS \
    --lora_rank $LORA_RANK \
    --port $PORT \
    --device "$DEVICE" \
    --load_8bit

SERVER_EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $SERVER_EXIT_CODE -eq 0 ]; then
    echo "H-UAV server stopped gracefully"
else
    echo "H-UAV server exited with error (code: $SERVER_EXIT_CODE)"
fi
echo "============================================================"
