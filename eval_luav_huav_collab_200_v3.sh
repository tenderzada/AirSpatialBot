#!/bin/bash

# Evaluate L-UAV + H-UAV Hierarchical Collaboration V3
#
# L-UAV: 3-epoch LoRA (basic capability)
# H-UAV: 10-epoch LoRA (enhanced capability)
# Strategy: L-UAV requests H-UAV's answer directly
# Evaluation: First 200 samples with forced query every 5 samples

echo "============================================================"
echo "L-UAV (3-epoch) + H-UAV (10-epoch) Hierarchical Collaboration V3"
echo "============================================================"

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
LUAV_LORA_WEIGHTS="./outputs/lora_injection_sqa_3epoch/lora_adapters_epochfinal.pt"  # 3-epoch
TEST_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/comparison_luav_huav_collab_200_v3"

# Devices
LUAV_DEVICE="cuda:1"  # L-UAV on GPU 1

# LoRA config
LORA_RANK=8
TARGET_LAYERS="8 16 24"

# H-UAV connection
HUAV_ADDRESS="localhost:50052"  # V3 uses port 50052
CONFIDENCE_THRESHOLD=0.5
FORCE_HUAV_EVERY_N=2  # Force H-UAV query every 5 samples

# Data
MAX_SAMPLES=200  # First 200 samples

echo ""
echo "Configuration:"
echo "  L-UAV LoRA: $LUAV_LORA_WEIGHTS (3 epochs)"
echo "  H-UAV: $HUAV_ADDRESS (10-epoch LoRA)"
echo "  Test data: $TEST_DATA"
echo "  Samples: $MAX_SAMPLES"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Architecture V3:"
echo "  L-UAV (3-epoch LoRA) requests H-UAV's answer directly"
echo "  ↓"
echo "  L-UAV: Basic capability (3-epoch LoRA)"
echo "  ↓"
echo "  Request H-UAV's answer every 5 samples or when confidence < 0.5"
echo "  ↓"
echo "  H-UAV: Enhanced capability (10-epoch LoRA)"
echo "  ↓"
echo "  L-UAV uses H-UAV's answer when needed"
echo ""
echo "Strategy: Answer Delegation (not memory injection)"
echo ""
echo "Collaboration:"
echo "  Confidence threshold: $CONFIDENCE_THRESHOLD"
echo "  Forced H-UAV query: Every $FORCE_HUAV_EVERY_N samples"
echo ""
echo "============================================================"
echo ""

# Check L-UAV LoRA weights
if [ ! -f "$LUAV_LORA_WEIGHTS" ]; then
    echo "❌ L-UAV LoRA weights not found: $LUAV_LORA_WEIGHTS"
    echo ""
    echo "Please train L-UAV LoRA first (3 epochs):"
    echo "  ./train_lora_injection_sqa_3epoch.sh"
    echo ""
    exit 1
fi

# Check H-UAV connection
echo "Checking H-UAV server connection at $HUAV_ADDRESS..."
if ! timeout 3 bash -c "echo >/dev/tcp/${HUAV_ADDRESS%%:*}/${HUAV_ADDRESS##*:}" 2>/dev/null; then
    echo "⚠️  Warning: Cannot connect to H-UAV at $HUAV_ADDRESS"
    echo ""
    echo "Please start H-UAV server first:"
    echo "  ./start_huav_lora_answer_server_v3.sh"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Evaluation cancelled."
        exit 0
    fi
else
    echo "✓ H-UAV server is running"
fi

echo ""
echo "Starting L-UAV + H-UAV hierarchical collaboration evaluation V3..."
echo ""

python hierarchical_uav/eval_sqa_lora_socket_v3.py \
    --model_path "$MODEL_PATH" \
    --vision_tower "$VISION_TOWER" \
    --lora_weights "$LUAV_LORA_WEIGHTS" \
    --lora_rank $LORA_RANK \
    --target_layers $TARGET_LAYERS \
    --device "$LUAV_DEVICE" \
    --huav_address "$HUAV_ADDRESS" \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --force_huav_every_n $FORCE_HUAV_EVERY_N \
    --test_data "$TEST_DATA" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples $MAX_SAMPLES \
    --load_8bit

EVAL_EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "L-UAV + H-UAV Hierarchical Collaboration V3 Complete ✓"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/results.json"
    echo ""

    # Show quick metrics if jq is available
    if [ -f "$OUTPUT_DIR/results.json" ] && command -v jq &> /dev/null; then
        echo "Quick Metrics:"
        jq -r '.metrics | "  MAE:  \(.mae)\n  RMSE: \(.rmse)\n  MRE:  \(.mre)"' "$OUTPUT_DIR/results.json"
        echo ""
        jq -r '.collaboration_stats | "Collaboration Stats:\n  L-UAV LoRA: \(.luav_lora_epochs) epochs\n  H-UAV LoRA: \(.huav_lora_epochs) epochs\n  Strategy: \(.strategy)\n  H-UAV requests: \(.huav_requests)\n  Standalone: \(.standalone_inferences)\n  H-UAV usage: \(.huav_usage_rate * 100)%"' "$OUTPUT_DIR/results.json"
    fi
else
    echo "L-UAV + H-UAV Hierarchical Collaboration V3 Failed ✗"
fi
echo "============================================================"
