#!/bin/bash

# Start H-UAV LoRA Memory Server
#
# H-UAV 记忆编织器服务器：LoRA 适配器注入冻结的 LLaVA
# 接收 L-UAV 的请求，返回动态生成的记忆 tokens

echo "============================================================"
echo "H-UAV LoRA Memory Server"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
LORA_WEIGHTS="./outputs/lora_injection_sqa/lora_adapters_final.pt"
PORT=8000
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
echo "  Port: $PORT"
echo "  Device: $DEVICE"
echo ""
echo "Architecture:"
echo "  ┌─────────────────────────────────┐"
echo "  │  Frozen Base LLaVA (7B params)  │"
echo "  └─────────────────────────────────┘"
echo "           ↓"
echo "  ┌─────────────────────────────────┐"
echo "  │  LoRA Injection @ Layers 8,16,24│"
echo "  │  - Target: q_proj, v_proj       │"
echo "  │  - Trainable: 8.4M params       │"
echo "  └─────────────────────────────────┘"
echo "           ↓"
echo "  ┌─────────────────────────────────┐"
echo "  │  Memory Token Generator         │"
echo "  │  Output: [8, 4096]              │"
echo "  └─────────────────────────────────┘"
echo ""
echo "Endpoints:"
echo "  POST /get_memory"
echo "    - Input: L-UAV hidden states or image+question"
echo "    - Output: memory tokens [8, 4096]"
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

# Start server
echo "Starting H-UAV server..."
echo ""

python hierarchical_uav/huav_lora_server_v2.py \
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
