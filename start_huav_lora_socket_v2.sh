#!/bin/bash

# Start H-UAV LoRA Memory Server V2 (Socket-based)
#
# H-UAV使用10-epoch LoRA生成增强的memory tokens

echo "============================================================"
echo "Starting H-UAV LoRA Memory Server V2 (10-epoch LoRA)"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
LORA_WEIGHTS="./outputs/lora_injection_sqa/lora_adapters_best.pt"  # 10-epoch best checkpoint
DEVICE="cuda:0"

# LoRA config
LORA_RANK=8
TARGET_LAYERS="8 16 24"

# Server config
HOST="localhost"
PORT=50051

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS (10 epochs)"
echo "  Device: $DEVICE"
echo "  Server: $HOST:$PORT"
echo ""
echo "Architecture:"
echo "  H-UAV with 10-epoch LoRA (enhanced memory)"
echo "  ↓"
echo "  Provides enhanced memory tokens to L-UAV"
echo "  ↓"
echo "  L-UAV injects memory for better inference"
echo ""
echo "============================================================"
echo ""

# Check LoRA weights
if [ ! -f "$LORA_WEIGHTS" ]; then
    echo "❌ LoRA weights not found: $LORA_WEIGHTS"
    echo ""
    echo "Please train H-UAV LoRA first:"
    echo "  ./train_lora_injection_sqa.sh"
    echo ""
    exit 1
fi

echo "Starting H-UAV LoRA Memory Server V2..."
echo ""

python hierarchical_uav/huav_lora_server_socket_v2.py \
    --model_path "$MODEL_PATH" \
    --vision_tower "$VISION_TOWER" \
    --lora_weights "$LORA_WEIGHTS" \
    --lora_rank $LORA_RANK \
    --target_layers $TARGET_LAYERS \
    --host "$HOST" \
    --port $PORT \
    --device "$DEVICE" \
    --load_8bit
