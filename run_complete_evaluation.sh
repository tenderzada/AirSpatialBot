#!/bin/bash

# Complete Hierarchical UAV Evaluation Script
# This script runs the full evaluation pipeline:
# 1. Start H-UAV server with trained memory
# 2. Run L-UAV evaluation with forced memory queries
# 3. Analyze results

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "HIERARCHICAL UAV COMPLETE EVALUATION PIPELINE"
echo "================================================================================"
echo ""

# Configuration
CHECKPOINT_DIR="./outputs/huav_training"
CHECKPOINT_FILE="$CHECKPOINT_DIR/huav_memory_final.pt"
OUTPUT_DIR="./outputs/hierarchical_uav"
GROUND_TRUTH="/mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo -e "${RED}✗ Checkpoint not found: $CHECKPOINT_FILE${NC}"
    echo ""
    echo "Please ensure training was completed successfully."
    echo "Expected checkpoint location: $CHECKPOINT_FILE"
    exit 1
fi

echo -e "${GREEN}✓ Found trained checkpoint: $CHECKPOINT_FILE${NC}"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Step 1: Start H-UAV Server
echo "================================================================================"
echo "STEP 1: Starting H-UAV Server"
echo "================================================================================"
echo ""
echo "H-UAV will:"
echo "  - Load trained MAC memory from checkpoint"
echo "  - Start gRPC server on port 50051"
echo "  - Wait for L-UAV connections"
echo ""
echo -e "${YELLOW}NOTE: H-UAV will run in the background.${NC}"
echo "      You can monitor it with: tail -f h_uav_server.log"
echo ""

# Set environment variable for checkpoint
export LOAD_MEMORY="$CHECKPOINT_FILE"

# Start H-UAV in background and log output
echo "Starting H-UAV server..."
nohup bash -c "
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cuda:0 \
    --port 50051 \
    --model_path /mnt/data/AirSpatialBot \
    --vision_tower /mnt/data/clip-vit-large-patch14-336 \
    --test_data /mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl \
    --image_dir /mnt/data/AirSpatial/images \
    --load_8bit \
    --load-memory $LOAD_MEMORY \
    --output $OUTPUT_DIR/task1_h-uav_results.jsonl
" > h_uav_server.log 2>&1 &

HUAV_PID=$!
echo "H-UAV server PID: $HUAV_PID"
echo $HUAV_PID > h_uav.pid

# Wait for H-UAV to start (loading 8-bit model + MAC takes time)
echo "Waiting for H-UAV to initialize (this may take 30-60 seconds)..."
sleep 20

# Check if H-UAV process is still running
if ! kill -0 $HUAV_PID 2>/dev/null; then
    echo -e "${RED}✗ H-UAV process died during startup!${NC}"
    echo "Check h_uav_server.log for errors:"
    tail -20 h_uav_server.log
    exit 1
fi

# Wait for port to open (check every 5 seconds for up to 60 seconds)
echo "Waiting for H-UAV server to start listening on port 50051..."
MAX_WAIT=60
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if nc -z localhost 50051 2>/dev/null; then
        echo -e "${GREEN}✓ H-UAV server is running and accepting connections${NC}"
        break
    fi

    # Check if process is still alive
    if ! kill -0 $HUAV_PID 2>/dev/null; then
        echo -e "${RED}✗ H-UAV process died!${NC}"
        echo "Last 30 lines of h_uav_server.log:"
        tail -30 h_uav_server.log
        exit 1
    fi

    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  Still waiting... (${ELAPSED}s/${MAX_WAIT}s)"
done

# Final check
if ! nc -z localhost 50051 2>/dev/null; then
    echo -e "${RED}✗ H-UAV server did not start within ${MAX_WAIT} seconds${NC}"
    echo ""
    echo "Last 30 lines of h_uav_server.log:"
    tail -30 h_uav_server.log
    echo ""
    echo "The server process is still running (PID: $HUAV_PID)."
    echo "You can:"
    echo "  1. Wait longer and check: nc -z localhost 50051"
    echo "  2. Monitor the log: tail -f h_uav_server.log"
    echo "  3. Kill the process: kill $HUAV_PID"
    kill $HUAV_PID 2>/dev/null || true
    exit 1
fi

echo ""

# Step 2: Run L-UAV with forced memory queries
echo "================================================================================"
echo "STEP 2: Running L-UAV Evaluation"
echo "================================================================================"
echo ""
echo "L-UAV will:"
echo "  - Connect to H-UAV server at localhost:50051"
echo "  - Force 30% of queries to use H-UAV memory"
echo "  - Evaluate all test samples"
echo ""

python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device cuda:1 \
    --huav_address localhost:50051 \
    --threshold 0.7 \
    --force-query-rate 0.3 \
    --model_path /mnt/data/AirSpatialBot \
    --vision_tower /mnt/data/clip-vit-large-patch14-336 \
    --test_data /mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl \
    --image_dir /mnt/data/AirSpatial/images \
    --load_8bit \
    --output $OUTPUT_DIR/task1_l-uav_results.jsonl

LUAV_EXIT=$?

# Stop H-UAV server
echo ""
echo "Stopping H-UAV server..."
kill $HUAV_PID 2>/dev/null || true
rm -f h_uav.pid

if [ $LUAV_EXIT -ne 0 ]; then
    echo -e "${RED}✗ L-UAV evaluation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ L-UAV evaluation completed${NC}"
echo ""

# Step 3: Analyze Results
echo "================================================================================"
echo "STEP 3: Analyzing Results"
echo "================================================================================"
echo ""

python analyze_uav_results.py \
    --luav-results $OUTPUT_DIR/task1_l-uav_results.jsonl \
    --ground-truth $GROUND_TRUTH \
    --output-report $OUTPUT_DIR/evaluation_report.json

echo ""
echo "================================================================================"
echo "EVALUATION COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - L-UAV results: $OUTPUT_DIR/task1_l-uav_results.jsonl"
echo "  - Analysis report: $OUTPUT_DIR/evaluation_report.json"
echo "  - H-UAV log: h_uav_server.log"
echo ""
echo "Next steps:"
echo "  1. Review the analysis report above"
echo "  2. Check detailed report: cat $OUTPUT_DIR/evaluation_report.json | jq '.'"
echo "  3. If memory usage is still 0%, check h_uav_server.log for connection issues"
echo ""
