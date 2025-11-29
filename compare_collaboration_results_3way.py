#!/usr/bin/env python3
"""
Compare 3-way collaboration experiment results (200 samples)

Experiments:
1. H-UAV Baseline (WITHOUT LoRA MemGen)
2. H-UAV Standalone (WITH LoRA MemGen, 10 epochs)
3. L-UAV + H-UAV Collaboration

Usage:
    python compare_collaboration_results_3way.py
"""

import json
import sys
from pathlib import Path

def load_results(path):
    """Load results JSON file."""
    with open(path) as f:
        return json.load(f)

def print_3way_comparison():
    """Print 3-way comparison of all experiments."""

    baseline_path = Path("./outputs/comparison_huav_baseline_200/results.json")
    huav_path = Path("./outputs/comparison_huav_standalone_200/results.json")
    collab_path = Path("./outputs/comparison_luav_huav_collab_200/results.json")

    # Check if all files exist
    missing = []
    if not baseline_path.exists():
        missing.append(("H-UAV Baseline", baseline_path, "./eval_huav_baseline_200.sh"))
    if not huav_path.exists():
        missing.append(("H-UAV Standalone", huav_path, "./eval_huav_standalone_200.sh"))
    if not collab_path.exists():
        missing.append(("L-UAV+H-UAV Collab", collab_path, "./eval_luav_huav_collab_200.sh"))

    if missing:
        print("❌ Missing results:")
        for name, path, script in missing:
            print(f"   - {name}: {path}")
            print(f"     Run: {script}")
        return False

    # Load results
    baseline = load_results(baseline_path)
    huav = load_results(huav_path)
    collab = load_results(collab_path)

    # Extract metrics
    baseline_metrics = baseline['metrics']
    huav_metrics = huav['metrics']
    collab_metrics = collab['metrics']

    # Print header
    print("=" * 110)
    print(" " * 30 + "3-Way Collaboration Comparison (200 samples)")
    print("=" * 110)
    print()

    # Overall comparison
    print("📊 Overall Metrics Comparison:")
    print("=" * 110)
    print(f"{'Metric':<10} {'Baseline':<20} {'H-UAV+MemGen':<20} {'L-UAV+H-UAV':<20} {'MemGen Gain':<20} {'Collab Gain':<15}")
    print("-" * 110)

    for metric_name in ['mae', 'rmse', 'mre']:
        baseline_val = baseline_metrics[metric_name]
        huav_val = huav_metrics[metric_name]
        collab_val = collab_metrics[metric_name]

        # Calculate improvements
        memgen_gain = ((baseline_val - huav_val) / baseline_val) * 100  # positive = better
        collab_gain = ((baseline_val - collab_val) / baseline_val) * 100  # positive = better

        # Format values
        if metric_name == 'mre':
            baseline_str = f"{baseline_val*100:.2f}%"
            huav_str = f"{huav_val*100:.2f}%"
            collab_str = f"{collab_val*100:.2f}%"
        else:
            baseline_str = f"{baseline_val:.2f}"
            huav_str = f"{huav_val:.2f}"
            collab_str = f"{collab_val:.2f}"

        memgen_str = f"{memgen_gain:+.1f}%"
        collab_str_gain = f"{collab_gain:+.1f}%"

        print(f"{metric_name.upper():<10} {baseline_str:<20} {huav_str:<20} {collab_str:<20} {memgen_str:<20} {collab_str_gain:<15}")

    print("=" * 110)
    print("Note: Positive gain means improvement over baseline")
    print()

    # Collaboration statistics
    if 'collaboration_stats' in collab:
        collab_stats = collab['collaboration_stats']
        print("🤝 Collaboration Statistics:")
        print("-" * 110)
        print(f"  H-UAV requests:        {collab_stats.get('huav_requests', 0)}")
        print(f"  Standalone inferences: {collab_stats.get('standalone_inferences', 0)}")
        print(f"  H-UAV usage rate:      {collab_stats.get('huav_usage_rate', 0)*100:.1f}%")
        print()

    # Per-type comparison
    print("📋 Per-Type Metrics (MRE) Comparison:")
    print("=" * 110)

    baseline_type = baseline.get('type_metrics', {})
    huav_type = huav.get('type_metrics', {})
    collab_type = collab.get('type_metrics', {})

    all_types = sorted(set(list(baseline_type.keys()) + list(huav_type.keys()) + list(collab_type.keys())))

    print(f"{'Type':<12} {'Baseline':<18} {'H-UAV+MemGen':<18} {'L-UAV+H-UAV':<18} {'MemGen Gain':<18} {'Collab Gain':<15}")
    print("-" * 110)

    for qtype in all_types:
        baseline_mre = baseline_type.get(qtype, {}).get('mre', 0)
        huav_mre = huav_type.get(qtype, {}).get('mre', 0)
        collab_mre = collab_type.get(qtype, {}).get('mre', 0)

        memgen_gain = ((baseline_mre - huav_mre) / baseline_mre) * 100 if baseline_mre > 0 else 0
        collab_gain = ((baseline_mre - collab_mre) / baseline_mre) * 100 if baseline_mre > 0 else 0

        baseline_str = f"{baseline_mre*100:.2f}%"
        huav_str = f"{huav_mre*100:.2f}%"
        collab_str = f"{collab_mre*100:.2f}%"
        memgen_str = f"{memgen_gain:+.1f}%"
        collab_str_gain = f"{collab_gain:+.1f}%"

        print(f"{qtype:<12} {baseline_str:<18} {huav_str:<18} {collab_str:<18} {memgen_str:<18} {collab_str_gain:<15}")

    print("=" * 110)
    print()

    # Detailed analysis
    print("💡 Detailed Analysis:")
    print("-" * 110)

    mae_memgen_gain = ((baseline_metrics['mae'] - huav_metrics['mae']) / baseline_metrics['mae']) * 100
    mae_collab_gain = ((baseline_metrics['mae'] - collab_metrics['mae']) / baseline_metrics['mae']) * 100
    mre_memgen_gain = ((baseline_metrics['mre'] - huav_metrics['mre']) / baseline_metrics['mre']) * 100
    mre_collab_gain = ((baseline_metrics['mre'] - collab_metrics['mre']) / baseline_metrics['mre']) * 100

    print()
    print("1️⃣  Baseline → H-UAV with MemGen:")
    if mae_memgen_gain > 0:
        print(f"   ✅ MemGen IMPROVES performance:")
        print(f"      - MAE reduced by {mae_memgen_gain:.2f}%")
        print(f"      - MRE reduced by {mre_memgen_gain:.2f}%")
        print(f"   → LoRA-enhanced memory generation is effective")
    else:
        print(f"   ⚠️  MemGen shows negative impact:")
        print(f"      - MAE increased by {abs(mae_memgen_gain):.2f}%")
        print(f"   → May need to adjust LoRA training")

    print()
    print("2️⃣  Baseline → L-UAV + H-UAV Collaboration:")
    if mae_collab_gain > 0:
        print(f"   ✅ Collaboration IMPROVES performance:")
        print(f"      - MAE reduced by {mae_collab_gain:.2f}%")
        print(f"      - MRE reduced by {mre_collab_gain:.2f}%")
        print(f"   → L-UAV benefits from H-UAV memory guidance")
    else:
        print(f"   ⚠️  Collaboration shows negative impact:")
        print(f"      - MAE increased by {abs(mae_collab_gain):.2f}%")
        print(f"   → Memory integration needs improvement")

    print()
    print("3️⃣  H-UAV with MemGen vs L-UAV + H-UAV:")
    huav_vs_collab = ((huav_metrics['mae'] - collab_metrics['mae']) / huav_metrics['mae']) * 100
    if abs(huav_vs_collab) < 2:
        print(f"   ➡️  Similar performance (difference: {huav_vs_collab:+.1f}%)")
        print(f"   → Both approaches leverage MemGen effectively")
    elif huav_vs_collab < 0:
        print(f"   ✅ Collaboration outperforms standalone by {abs(huav_vs_collab):.2f}%")
        print(f"   → Collaboration mechanism adds value beyond MemGen alone")
    else:
        print(f"   ⚠️  Standalone outperforms collaboration by {huav_vs_collab:.2f}%")
        print(f"   → Direct MemGen usage more effective than collaboration")

    print()
    print("=" * 110)

    # Summary table
    print()
    print("📈 Summary Table:")
    print("=" * 110)
    print(f"{'Approach':<30} {'MAE':<15} {'MRE':<15} {'vs Baseline':<20} {'Rank':<10}")
    print("-" * 110)

    results = [
        ("Baseline (no MemGen)", baseline_metrics['mae'], baseline_metrics['mre'], "0.0%", "-"),
        ("H-UAV + MemGen", huav_metrics['mae'], huav_metrics['mre'], f"{mae_memgen_gain:+.1f}%", ""),
        ("L-UAV + H-UAV", collab_metrics['mae'], collab_metrics['mre'], f"{mae_collab_gain:+.1f}%", "")
    ]

    # Sort by MAE to determine rank
    sorted_results = sorted(results[1:], key=lambda x: x[1])
    for i, (name, mae, mre, gain, _) in enumerate(sorted_results):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        mae_str = f"{mae:.2f}"
        mre_str = f"{mre*100:.2f}%"
        print(f"{name:<30} {mae_str:<15} {mre_str:<15} {gain:<20} {rank:<10}")

    # Print baseline
    name, mae, mre, gain, _ = results[0]
    mae_str = f"{mae:.2f}"
    mre_str = f"{mre*100:.2f}%"
    print(f"{name:<30} {mae_str:<15} {mre_str:<15} {gain:<20} {'Baseline':<10}")

    print("=" * 110)

    # Key findings
    print()
    print("🔑 Key Findings:")
    print("-" * 110)
    print(f"1. Baseline Performance (no MemGen):")
    print(f"   MAE: {baseline_metrics['mae']:.2f}, MRE: {baseline_metrics['mre']*100:.2f}%")
    print()
    print(f"2. H-UAV with LoRA MemGen (10 epochs):")
    print(f"   MAE: {huav_metrics['mae']:.2f}, MRE: {huav_metrics['mre']*100:.2f}%")
    print(f"   Improvement: {mae_memgen_gain:+.1f}% MAE, {mre_memgen_gain:+.1f}% MRE")
    print()
    print(f"3. L-UAV + H-UAV Collaboration:")
    print(f"   MAE: {collab_metrics['mae']:.2f}, MRE: {collab_metrics['mre']*100:.2f}%")
    if 'collaboration_stats' in collab:
        print(f"   H-UAV usage: {collab_stats.get('huav_usage_rate', 0)*100:.1f}%")
    print(f"   Improvement: {mae_collab_gain:+.1f}% MAE, {mre_collab_gain:+.1f}% MRE")
    print()

    # Recommendation
    best_approach = sorted_results[0][0]
    print(f"💡 Recommendation:")
    print(f"   Best approach: {best_approach}")
    if mae_memgen_gain > 5:
        print(f"   ✅ LoRA MemGen provides significant improvement ({mae_memgen_gain:.1f}%)")
    if mae_collab_gain > mae_memgen_gain:
        print(f"   ✅ Collaboration outperforms standalone MemGen")
    print("=" * 110)

    return True


if __name__ == '__main__':
    success = print_3way_comparison()
    sys.exit(0 if success else 1)
