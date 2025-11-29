#!/bin/bash

# Run 3-way Hierarchical UAV Comparison V3 (Answer Delegation Strategy)
#
# Experiment 1: H-UAV Baseline (NO LoRA) - 200 samples
# Experiment 2: H-UAV + MemGen (10-epoch LoRA) - 200 samples
# Experiment 3: L-UAV (3-epoch) + H-UAV (10-epoch) Hierarchical Collaboration V3 - 200 samples
#
# V3 Strategy: L-UAV requests H-UAV's answer directly (not memory tokens)

echo "========================================================================"
echo "3-Way Hierarchical UAV Comparison V3 (Answer Delegation Strategy)"
echo "========================================================================"
echo ""
echo "Experiments:"
echo "  1. H-UAV Baseline (NO LoRA) - Establish baseline performance"
echo "  2. H-UAV + MemGen (10-epoch LoRA) - Test LoRA memory enhancement"
echo "  3. L-UAV (3-epoch) + H-UAV (10-epoch) V3 - Test hierarchical collaboration"
echo ""
echo "Each experiment: 200 samples"
echo ""
echo "V3 Strategy:"
echo "  - L-UAV has basic capability (3-epoch LoRA)"
echo "  - L-UAV requests H-UAV's answer directly when needed"
echo "  - H-UAV provides enhanced answer using 10-epoch LoRA"
echo "  - Simpler and more effective than memory injection"
echo ""
echo "Expected outcomes:"
echo "  - Experiment 2 should show improvement over Experiment 1 (LoRA benefit)"
echo "  - Experiment 3 should show L-UAV benefits from H-UAV's enhanced answers"
echo "  - V3 should actually show different results (not identical to baseline)"
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
read -p "Continue to Experiment 3 V3 (requires H-UAV Answer Server V3)? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Experiments stopped. You can run Experiment 3 later with:"
    echo "  ./eval_luav_huav_collab_200_v3.sh"
    exit 0
fi

# Check if H-UAV server needs to be started
echo ""
echo "Checking H-UAV Answer Server V3..."
if ! timeout 3 bash -c "echo >/dev/tcp/localhost/50052" 2>/dev/null; then
    echo "⚠️  H-UAV Answer Server V3 not running (port 50052)"
    echo ""
    echo "Please start H-UAV Answer Server V3 in a separate terminal:"
    echo "  ./start_huav_lora_answer_server_v3.sh"
    echo ""
    read -p "Press Enter when H-UAV Answer Server V3 is ready..."
else
    echo "✓ H-UAV Answer Server V3 is running"
fi

# Experiment 3: L-UAV + H-UAV Hierarchical Collaboration V3
echo ""
echo "========================================================================"
echo "Experiment 3/3: L-UAV (3-epoch) + H-UAV (10-epoch) Hierarchical V3"
echo "========================================================================"
echo ""

if [ -f "./eval_luav_huav_collab_200_v3.sh" ]; then
    ./eval_luav_huav_collab_200_v3.sh
    EXP3_EXIT=$?
else
    echo "❌ Script not found: eval_luav_huav_collab_200_v3.sh"
    EXP3_EXIT=1
fi

if [ $EXP3_EXIT -ne 0 ]; then
    echo "❌ Experiment 3 V3 failed."
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

# Check which collaboration version we have
if [ -f "./outputs/comparison_luav_huav_collab_200_v3/results.json" ]; then
    echo "Using V3 results (Answer Delegation)"
    python -c "
import json

# Load results
baseline = json.load(open('./outputs/comparison_huav_baseline_200/results.json'))
huav_memgen = json.load(open('./outputs/comparison_huav_standalone_200/results.json'))
collab_v3 = json.load(open('./outputs/comparison_luav_huav_collab_200_v3/results.json'))

print('='*80)
print('3-Way Comparison Results (V3 - Answer Delegation)')
print('='*80)
print()
print(f\"Experiment 1 (Baseline):     MAE={baseline['metrics']['mae']:.4f}\")
print(f\"Experiment 2 (H-UAV+MemGen): MAE={huav_memgen['metrics']['mae']:.4f}\")
print(f\"Experiment 3 (L-UAV+H-UAV):  MAE={collab_v3['metrics']['mae']:.4f}\")
print()

# Calculate gains
baseline_mae = baseline['metrics']['mae']
huav_gain = ((baseline_mae - huav_memgen['metrics']['mae']) / baseline_mae) * 100
collab_gain = ((baseline_mae - collab_v3['metrics']['mae']) / baseline_mae) * 100

print('Performance Gains vs Baseline:')
print(f\"  H-UAV+MemGen:     {huav_gain:+.2f}%\")
print(f\"  L-UAV+H-UAV V3:   {collab_gain:+.2f}%\")
print()

# Check if results are different
if abs(baseline_mae - collab_v3['metrics']['mae']) < 0.001:
    print('⚠️  WARNING: L-UAV+H-UAV V3 results are identical to baseline!')
    print('   This suggests collaboration is not working.')
else:
    print('✓ Results are different - collaboration appears to be working!')
print()
print('='*80)
"
else
    echo "⚠️  V3 results not found"
fi

echo ""
echo "========================================================================"
echo "3-Way Comparison Complete"
echo "========================================================================"
echo ""
echo "Results:"
echo "  - Experiment 1 (Baseline): ./outputs/comparison_huav_baseline_200/results.json"
echo "  - Experiment 2 (H-UAV+MemGen): ./outputs/comparison_huav_standalone_200/results.json"
echo "  - Experiment 3 (L-UAV+H-UAV V3): ./outputs/comparison_luav_huav_collab_200_v3/results.json"
echo ""
