#!/bin/bash

# Run Complete 3-Way Collaboration Comparison Experiment
#
# 一键运行所有对比实验（200 samples）

echo "============================================================"
echo "     3-Way Collaboration Comparison Experiment Suite"
echo "============================================================"
echo ""
echo "This will run:"
echo "  1. H-UAV Baseline WITHOUT MemGen (200 samples)"
echo "  2. H-UAV Standalone WITH LoRA MemGen (200 samples)"
echo "  3. L-UAV + H-UAV Collaboration (200 samples)"
echo "  4. Compare and analyze all results"
echo ""
echo "Estimated time: ~45-60 minutes total"
echo "============================================================"
echo ""

read -p "Start complete 3-way comparison experiment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Experiment cancelled."
    exit 0
fi

# Check H-UAV server is running
echo ""
echo "Checking H-UAV server..."
if timeout 2 bash -c "echo > /dev/tcp/localhost/50051" 2>/dev/null; then
    echo "✓ H-UAV server is running"
else
    echo ""
    echo "❌ H-UAV server not running!"
    echo ""
    echo "Please start H-UAV server first:"
    echo "  ./start_huav_lora_socket.sh"
    echo ""
    exit 1
fi

echo ""
echo "============================================================"
echo "Experiment 1/3: H-UAV Baseline (WITHOUT MemGen)"
echo "============================================================"
echo ""

./eval_huav_baseline_200.sh

BASELINE_EXIT=$?

if [ $BASELINE_EXIT -ne 0 ]; then
    echo ""
    echo "❌ H-UAV baseline evaluation failed!"
    echo "Please check error messages above."
    exit 1
fi

echo ""
echo "✅ Experiment 1/3 completed"
echo ""
echo "Press Enter to continue to Experiment 2..."
read

echo ""
echo "============================================================"
echo "Experiment 2/3: H-UAV Standalone with LoRA MemGen"
echo "============================================================"
echo ""

./eval_huav_standalone_200.sh

HUAV_EXIT=$?

if [ $HUAV_EXIT -ne 0 ]; then
    echo ""
    echo "❌ H-UAV standalone evaluation failed!"
    echo "Please check error messages above."
    exit 1
fi

echo ""
echo "✅ Experiment 2/3 completed"
echo ""
echo "Press Enter to continue to Experiment 3..."
read

echo ""
echo "============================================================"
echo "Experiment 3/3: L-UAV + H-UAV Collaboration"
echo "============================================================"
echo ""

./eval_luav_huav_collab_200.sh

COLLAB_EXIT=$?

if [ $COLLAB_EXIT -ne 0 ]; then
    echo ""
    echo "❌ L-UAV+H-UAV collaboration evaluation failed!"
    echo "Please check error messages above."
    exit 1
fi

echo ""
echo "✅ Experiment 3/3 completed"
echo ""

echo ""
echo "============================================================"
echo "Generating 3-Way Comparison Report"
echo "============================================================"
echo ""

python compare_collaboration_results_3way.py

COMPARE_EXIT=$?

echo ""
echo "============================================================"
echo "Experiment Suite Complete!"
echo "============================================================"
echo ""

if [ $COMPARE_EXIT -eq 0 ]; then
    echo "✅ All experiments completed successfully"
    echo ""
    echo "Results:"
    echo "  - Baseline (no MemGen):  ./outputs/comparison_huav_baseline_200/results.json"
    echo "  - H-UAV + MemGen:        ./outputs/comparison_huav_standalone_200/results.json"
    echo "  - L-UAV + H-UAV Collab:  ./outputs/comparison_luav_huav_collab_200/results.json"
    echo ""
    echo "View 3-way comparison:"
    echo "  python compare_collaboration_results_3way.py"
    echo ""
else
    echo "⚠️  Comparison script encountered an issue"
fi

echo "============================================================"
