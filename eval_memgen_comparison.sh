#!/bin/bash

# Evaluation script: Compare H-UAV with/without MemGen
# Evaluates on first 500 samples

echo "=========================================="
echo "MemGen Comparison Evaluation"
echo "=========================================="

# Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
EVAL_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
MEMGEN_CHECKPOINT="./outputs/memgen_weaver_sqa/best_checkpoint"
OUTPUT_DIR="./outputs/memgen_eval_comparison"
MAX_SAMPLES=500

# Check if checkpoint exists
if [ ! -d "$MEMGEN_CHECKPOINT" ]; then
    echo "ERROR: MemGen checkpoint not found at $MEMGEN_CHECKPOINT"
    echo "Please train MemGen first using: ./train_memgen_weaver.sh"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Evaluation Data: $EVAL_DATA"
echo "  MemGen Checkpoint: $MEMGEN_CHECKPOINT"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Starting evaluation..."
echo "=========================================="

# Run evaluation
python hierarchical_uav/eval_memgen_comparison.py \
    --model_path $MODEL_PATH \
    --eval_data $EVAL_DATA \
    --memgen_checkpoint $MEMGEN_CHECKPOINT \
    --max_samples $MAX_SAMPLES \
    --output_dir $OUTPUT_DIR

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo "  - comparison_results.json: Detailed results"
echo "  - memgen_comparison.png: Comparison charts"
echo ""
echo "Summary:"
cat $OUTPUT_DIR/comparison_results.json | grep -A 5 '"stats"'
echo "=========================================="
