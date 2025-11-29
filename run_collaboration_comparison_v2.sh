#!/bin/bash

# Run 3-way Collaboration Comparison Experiments (V2 - LoRA-based)
#
# Experiment 1: H-UAV Baseline (NO LoRA) - 200 samples
# Experiment 2: H-UAV + MemGen (10-epoch LoRA) - 200 samples
# Experiment 3: L-UAV (3-epoch LoRA) + H-UAV (10-epoch LoRA) Collaboration - 200 samples

echo "========================================================================"
echo "3-Way Collaboration Comparison (V2 - LoRA-based)"
echo "========================================================================"
echo ""
echo "Experiments:"
echo "  1. H-UAV Baseline (NO LoRA) - Establish baseline performance"
echo "  2. H-UAV + MemGen (10-epoch LoRA) - Test LoRA memory enhancement"
echo "  3. L-UAV (3-epoch) + H-UAV (10-epoch) - Test hierarchical collaboration"
echo ""
echo "Each experiment: 200 samples"
echo ""
echo "Expected outcomes:"
echo "  - Experiment 2 should show improvement over Experiment 1 (LoRA benefit)"
echo "  - Experiment 3 should show L-UAV benefits from H-UAV's enhanced memory"
echo ""
echo "========================================================================"
echo ""

read -p "Start all 3 experiments? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Experiments cancelled."
    exit 0
fi

# Experiment 1: Baseline (NO LoRA)
echo ""
echo "========================================================================"
echo "Experiment 1/3: H-UAV Baseline (NO LoRA) - 200 samples"
echo "========================================================================"
echo ""

if [ -f "./eval_huav_baseline_200.sh" ]; then
    ./eval_huav_baseline_200.sh
    EXP1_EXIT=$?
else
    echo "❌ Script not found: eval_huav_baseline_200.sh"
    EXP1_EXIT=1
fi

if [ $EXP1_EXIT -ne 0 ]; then
    echo "❌ Experiment 1 failed. Stopping."
    exit 1
fi

echo ""
echo "✓ Experiment 1/3 completed"
echo ""
read -p "Continue to Experiment 2? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Experiments stopped."
    exit 0
fi

# Experiment 2: H-UAV + MemGen (10-epoch LoRA)
echo ""
echo "========================================================================"
echo "Experiment 2/3: H-UAV + MemGen (10-epoch LoRA) - 200 samples"
echo "========================================================================"
echo ""

if [ -f "./eval_huav_standalone_200.sh" ]; then
    ./eval_huav_standalone_200.sh
    EXP2_EXIT=$?
else
    echo "❌ Script not found: eval_huav_standalone_200.sh"
    EXP2_EXIT=1
fi

if [ $EXP2_EXIT -ne 0 ]; then
    echo "❌ Experiment 2 failed. Stopping."
    exit 1
fi

echo ""
echo "✓ Experiment 2/3 completed"
echo ""
read -p "Continue to Experiment 3 (requires H-UAV server)? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Experiments stopped. You can run Experiment 3 later with:"
    echo "  ./eval_luav_huav_collab_200_v2.sh"
    exit 0
fi

# Check if H-UAV server needs to be started
echo ""
echo "Checking H-UAV server..."
if ! timeout 3 bash -c "echo >/dev/tcp/localhost/50051" 2>/dev/null; then
    echo "⚠️  H-UAV server not running"
    echo ""
    echo "Please start H-UAV server in a separate terminal:"
    echo "  ./start_huav_lora_socket_v2.sh"
    echo ""
    read -p "Press Enter when H-UAV server is ready..."
else
    echo "✓ H-UAV server is running"
fi

# Experiment 3: L-UAV + H-UAV Collaboration
echo ""
echo "========================================================================"
echo "Experiment 3/3: L-UAV (3-epoch) + H-UAV (10-epoch) Collaboration"
echo "========================================================================"
echo ""

if [ -f "./eval_luav_huav_collab_200_v2.sh" ]; then
    ./eval_luav_huav_collab_200_v2.sh
    EXP3_EXIT=$?
else
    echo "❌ Script not found: eval_luav_huav_collab_200_v2.sh"
    EXP3_EXIT=1
fi

if [ $EXP3_EXIT -ne 0 ]; then
    echo "❌ Experiment 3 failed."
    exit 1
fi

echo ""
echo "✓ Experiment 3/3 completed"
echo ""

# Generate comparison report
echo ""
echo "========================================================================"
echo "Generating 3-Way Comparison Report"
echo "========================================================================"
echo ""

if [ -f "./compare_collaboration_results_3way.py" ]; then
    python compare_collaboration_results_3way.py
    COMPARE_EXIT=$?

    if [ $COMPARE_EXIT -eq 0 ]; then
        echo ""
        echo "✅ All experiments completed successfully!"
        echo ""
        echo "Results:"
        echo "  - Experiment 1 (Baseline): ./outputs/comparison_huav_baseline_200/results.json"
        echo "  - Experiment 2 (H-UAV+MemGen): ./outputs/comparison_huav_standalone_200/results.json"
        echo "  - Experiment 3 (L-UAV+H-UAV): ./outputs/comparison_luav_huav_collab_200_v2/results.json"
        echo ""
    else
        echo "⚠️  Comparison report generation failed"
    fi
else
    echo "⚠️  Comparison script not found: compare_collaboration_results_3way.py"
    echo ""
    echo "Results are saved separately:"
    echo "  - Experiment 1: ./outputs/comparison_huav_baseline_200/results.json"
    echo "  - Experiment 2: ./outputs/comparison_huav_standalone_200/results.json"
    echo "  - Experiment 3: ./outputs/comparison_luav_huav_collab_200_v2/results.json"
fi

echo ""
echo "========================================================================"
echo "3-Way Comparison Complete"
echo "========================================================================"
