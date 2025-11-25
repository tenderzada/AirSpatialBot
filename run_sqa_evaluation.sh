#!/bin/bash
# SQA Evaluation Runner for Hierarchical UAV System
# Usage: ./run_sqa_evaluation.sh [h-uav|l-uav] [options]

set -e  # Exit on error

# ==================== Configuration ====================

# Model paths
MODEL_PATH="./models/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"

# Data paths
SQA_TEST_DATA="./data/metadata/airspatial_sqa_test.jsonl"
IMAGE_DIR="./data/images"

# Output directory
OUTPUT_DIR="./outputs/hierarchical_uav/sqa"

# H-UAV Configuration
HUAV_PORT=50051
HUAV_DEVICE="cuda:0"
HUAV_MEMORY_PATH="./outputs/huav_training/huav_memory_final.pt"  # Optional: trained memory

# L-UAV Configuration
LUAV_DEVICE="cuda:1"
LUAV_THRESHOLD=0.7
HUAV_ADDRESS="localhost:50051"

# Evaluation options
MAX_SAMPLES=""  # Leave empty to evaluate all samples, or set to e.g., "100" for testing
LOAD_8BIT="--load_8bit"  # Use 8-bit quantization to save memory
STANDALONE=""  # Set to "--standalone" for baseline evaluation

# ==================== Functions ====================

print_usage() {
    cat << EOF
SQA Evaluation Runner for Hierarchical UAV System

Usage:
    $0 h-uav [options]     Start H-UAV server for SQA evaluation
    $0 l-uav [options]     Start L-UAV client for SQA evaluation
    $0 both [options]      Start both H-UAV and L-UAV in separate terminals (requires tmux)

Options:
    --no-8bit              Disable 8-bit quantization (use full precision)
    --standalone           Run L-UAV in standalone mode (no H-UAV connection)
    --max-samples N        Evaluate only N samples (for testing)
    --threshold T          Set self-matching threshold (default: 0.7)
    --port P               Set H-UAV port (default: 50051)

Examples:
    # Start H-UAV server
    $0 h-uav

    # Start L-UAV client (in another terminal)
    $0 l-uav

    # Run L-UAV standalone (baseline)
    $0 l-uav --standalone

    # Test with 100 samples
    $0 l-uav --max-samples 100

    # Run both in tmux (automated)
    $0 both
EOF
}

run_huav() {
    echo "=========================================="
    echo "Starting H-UAV Server for SQA Evaluation"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  Device: $HUAV_DEVICE"
    echo "  Port: $HUAV_PORT"
    echo "  Model: $MODEL_PATH"
    echo "  Vision Tower: $VISION_TOWER"
    if [ -f "$HUAV_MEMORY_PATH" ]; then
        echo "  Memory: $HUAV_MEMORY_PATH (trained)"
    else
        echo "  Memory: Random initialization (untrained)"
    fi
    echo ""

    mkdir -p "$OUTPUT_DIR"

    CMD="python hierarchical_uav/eval_sqa.py \
        --uav_type h-uav \
        --device $HUAV_DEVICE \
        --port $HUAV_PORT \
        --model_path $MODEL_PATH \
        --vision_tower $VISION_TOWER \
        --test_data $SQA_TEST_DATA \
        --image_dir $IMAGE_DIR \
        --output $OUTPUT_DIR/sqa_huav_results.jsonl \
        $LOAD_8BIT"

    if [ -f "$HUAV_MEMORY_PATH" ]; then
        CMD="$CMD --load-memory $HUAV_MEMORY_PATH"
    fi

    echo "Running: $CMD"
    echo ""
    eval $CMD
}

run_luav() {
    echo "=========================================="
    echo "Starting L-UAV Client for SQA Evaluation"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  Device: $LUAV_DEVICE"
    echo "  H-UAV Address: $HUAV_ADDRESS"
    echo "  Threshold: $LUAV_THRESHOLD"
    echo "  Model: $MODEL_PATH"
    echo "  Vision Tower: $VISION_TOWER"
    if [ -n "$STANDALONE" ]; then
        echo "  Mode: STANDALONE (no H-UAV connection)"
    else
        echo "  Mode: Client-Server (with H-UAV memory)"
    fi
    if [ -n "$MAX_SAMPLES" ]; then
        echo "  Max Samples: $MAX_SAMPLES"
    else
        echo "  Max Samples: All (17,526)"
    fi
    echo ""

    mkdir -p "$OUTPUT_DIR"

    CMD="python hierarchical_uav/eval_sqa.py \
        --uav_type l-uav \
        --device $LUAV_DEVICE \
        --huav_address $HUAV_ADDRESS \
        --threshold $LUAV_THRESHOLD \
        --model_path $MODEL_PATH \
        --vision_tower $VISION_TOWER \
        --test_data $SQA_TEST_DATA \
        --image_dir $IMAGE_DIR \
        --output $OUTPUT_DIR/sqa_luav_results.jsonl \
        $LOAD_8BIT \
        $STANDALONE"

    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi

    echo "Running: $CMD"
    echo ""
    eval $CMD
}

run_both() {
    # Check if tmux is available
    if ! command -v tmux &> /dev/null; then
        echo "Error: tmux is required to run both H-UAV and L-UAV"
        echo "Please install tmux or run H-UAV and L-UAV in separate terminals"
        exit 1
    fi

    echo "=========================================="
    echo "Starting Both H-UAV and L-UAV in tmux"
    echo "=========================================="
    echo ""
    echo "Creating tmux session 'sqa_eval'..."
    echo ""
    echo "Windows:"
    echo "  0: H-UAV Server"
    echo "  1: L-UAV Client"
    echo ""
    echo "To attach: tmux attach -t sqa_eval"
    echo "To switch: Ctrl+B, then 0 or 1"
    echo "To detach: Ctrl+B, then D"
    echo ""

    # Create new tmux session with H-UAV
    tmux new-session -d -s sqa_eval -n huav "bash -c './run_sqa_evaluation.sh h-uav; read -p \"Press Enter to close...\"'"

    # Create new window for L-UAV (wait 30s for H-UAV to start)
    tmux new-window -t sqa_eval:1 -n luav "bash -c 'echo \"Waiting 30s for H-UAV to start...\"; sleep 30; ./run_sqa_evaluation.sh l-uav; read -p \"Press Enter to close...\"'"

    echo "✓ tmux session created"
    echo ""
    echo "Attaching to session..."
    tmux attach -t sqa_eval
}

# ==================== Main ====================

# Parse arguments
MODE="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-8bit)
            LOAD_8BIT=""
            shift
            ;;
        --standalone)
            STANDALONE="--standalone"
            shift
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --threshold)
            LUAV_THRESHOLD="$2"
            shift 2
            ;;
        --port)
            HUAV_PORT="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Execute based on mode
case $MODE in
    h-uav)
        run_huav
        ;;
    l-uav)
        run_luav
        ;;
    both)
        run_both
        ;;
    "")
        echo "Error: Please specify mode (h-uav, l-uav, or both)"
        echo ""
        print_usage
        exit 1
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        echo ""
        print_usage
        exit 1
        ;;
esac
