#!/usr/bin/env python3
"""
Compare ablation study results: WITH LoRA vs WITHOUT LoRA

Usage:
    python compare_ablation_results.py
"""

import json
import sys
from pathlib import Path

def load_results(path):
    """Load results JSON file."""
    with open(path) as f:
        return json.load(f)

def print_comparison():
    """Print comparison of WITH vs WITHOUT LoRA."""

    with_lora_path = Path("./outputs/ablation_huav_with_lora/results.json")
    without_lora_path = Path("./outputs/ablation_huav_without_lora/results.json")

    # Check if both files exist
    if not with_lora_path.exists():
        print(f"❌ WITH LoRA results not found: {with_lora_path}")
        print("   Run: ./eval_huav_ablation_with_lora.sh")
        return False

    if not without_lora_path.exists():
        print(f"❌ WITHOUT LoRA results not found: {without_lora_path}")
        print("   Run: ./eval_huav_ablation_without_lora.sh")
        return False

    # Load results
    with_lora = load_results(with_lora_path)
    without_lora = load_results(without_lora_path)

    # Extract metrics
    with_metrics = with_lora['metrics']
    without_metrics = without_lora['metrics']

    # Print header
    print("=" * 80)
    print(" " * 20 + "H-UAV Ablation Study: LoRA Memory Weaver")
    print("=" * 80)
    print()

    # Overall comparison
    print("📊 Overall Metrics Comparison (100 samples):")
    print("=" * 80)
    print(f"{'Metric':<15} {'WITH LoRA':<20} {'WITHOUT LoRA':<20} {'Improvement':<20}")
    print("-" * 80)

    for metric_name in ['mae', 'rmse', 'mre']:
        with_val = with_metrics[metric_name]
        without_val = without_metrics[metric_name]

        # Calculate improvement
        if metric_name == 'mre':
            # MRE is percentage
            improvement = (without_val - with_val) * 100  # percentage points
            improvement_str = f"{improvement:+.2f} pp"
        else:
            # MAE, RMSE are absolute values (lower is better)
            improvement_pct = ((without_val - with_val) / without_val) * 100
            improvement_str = f"{improvement_pct:+.2f}%"

        # Format values
        if metric_name == 'mre':
            with_str = f"{with_val*100:.2f}%"
            without_str = f"{without_val*100:.2f}%"
        else:
            with_str = f"{with_val:.2f}"
            without_str = f"{without_val:.2f}"

        print(f"{metric_name.upper():<15} {with_str:<20} {without_str:<20} {improvement_str:<20}")

    print("=" * 80)
    print()

    # Per-type comparison
    print("📋 Per-Type Metrics (MRE):")
    print("=" * 80)

    with_type = with_lora.get('type_metrics', {})
    without_type = without_lora.get('type_metrics', {})

    all_types = sorted(set(list(with_type.keys()) + list(without_type.keys())))

    print(f"{'Question Type':<15} {'WITH LoRA':<20} {'WITHOUT LoRA':<20} {'Improvement':<20}")
    print("-" * 80)

    for qtype in all_types:
        with_mre = with_type.get(qtype, {}).get('mre', 0)
        without_mre = without_type.get(qtype, {}).get('mre', 0)

        improvement = (without_mre - with_mre) * 100

        with_str = f"{with_mre*100:.2f}%"
        without_str = f"{without_mre*100:.2f}%"
        improvement_str = f"{improvement:+.2f} pp"

        print(f"{qtype:<15} {with_str:<20} {without_str:<20} {improvement_str:<20}")

    print("=" * 80)
    print()

    # Summary
    print("💡 Summary:")
    print("-" * 80)

    mae_improvement_pct = ((without_metrics['mae'] - with_metrics['mae']) / without_metrics['mae']) * 100
    mre_improvement_pp = (without_metrics['mre'] - with_metrics['mre']) * 100

    if mae_improvement_pct < 0:
        print(f"✅ LoRA Memory Weaver IMPROVES performance:")
        print(f"   - MAE reduced by {abs(mae_improvement_pct):.2f}%")
        print(f"   - MRE reduced by {abs(mre_improvement_pp):.2f} percentage points")
    else:
        print(f"⚠️  LoRA Memory Weaver does NOT improve performance:")
        print(f"   - MAE increased by {mae_improvement_pct:.2f}%")
        print(f"   - MRE increased by {mre_improvement_pp:.2f} percentage points")

    print()
    print("Note: Lower MAE/RMSE/MRE is better")
    print("      Positive improvement means WITH LoRA performs better")
    print("=" * 80)

    return True


if __name__ == '__main__':
    success = print_comparison()
    sys.exit(0 if success else 1)
