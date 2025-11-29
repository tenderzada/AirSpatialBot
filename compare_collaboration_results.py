#!/usr/bin/env python3
"""
Compare collaboration experiment results (200 samples)

Experiments:
1. H-UAV Standalone with LoRA MemGen
2. L-UAV + H-UAV Collaboration

Usage:
    python compare_collaboration_results.py
"""

import json
import sys
from pathlib import Path

def load_results(path):
    """Load results JSON file."""
    with open(path) as f:
        return json.load(f)

def print_comparison():
    """Print comparison of H-UAV standalone vs L-UAV+H-UAV collaboration."""

    huav_path = Path("./outputs/comparison_huav_standalone_200/results.json")
    collab_path = Path("./outputs/comparison_luav_huav_collab_200/results.json")

    # Check if both files exist
    if not huav_path.exists():
        print(f"❌ H-UAV standalone results not found: {huav_path}")
        print("   Run: ./eval_huav_standalone_200.sh")
        return False

    if not collab_path.exists():
        print(f"❌ L-UAV+H-UAV collaboration results not found: {collab_path}")
        print("   Run: ./eval_luav_huav_collab_200.sh")
        return False

    # Load results
    huav = load_results(huav_path)
    collab = load_results(collab_path)

    # Extract metrics
    huav_metrics = huav['metrics']
    collab_metrics = collab['metrics']

    # Print header
    print("=" * 90)
    print(" " * 20 + "Collaboration Comparison: H-UAV vs L-UAV+H-UAV (200 samples)")
    print("=" * 90)
    print()

    # Overall comparison
    print("📊 Overall Metrics Comparison:")
    print("=" * 90)
    print(f"{'Metric':<15} {'H-UAV Standalone':<25} {'L-UAV+H-UAV':<25} {'Difference':<20}")
    print("-" * 90)

    for metric_name in ['mae', 'rmse', 'mre']:
        huav_val = huav_metrics[metric_name]
        collab_val = collab_metrics[metric_name]

        # Calculate difference (lower is better)
        diff = collab_val - huav_val
        diff_pct = (diff / huav_val) * 100 if huav_val != 0 else 0

        # Format values
        if metric_name == 'mre':
            huav_str = f"{huav_val*100:.2f}%"
            collab_str = f"{collab_val*100:.2f}%"
            diff_str = f"{diff*100:+.2f} pp ({diff_pct:+.1f}%)"
        else:
            huav_str = f"{huav_val:.2f}"
            collab_str = f"{collab_val:.2f}"
            diff_str = f"{diff:+.2f} ({diff_pct:+.1f}%)"

        print(f"{metric_name.upper():<15} {huav_str:<25} {collab_str:<25} {diff_str:<20}")

    print("=" * 90)
    print()

    # Collaboration statistics
    if 'collaboration_stats' in collab:
        collab_stats = collab['collaboration_stats']
        print("🤝 Collaboration Statistics:")
        print("-" * 90)
        print(f"  H-UAV requests:        {collab_stats.get('huav_requests', 0)}")
        print(f"  Standalone inferences: {collab_stats.get('standalone_inferences', 0)}")
        print(f"  H-UAV usage rate:      {collab_stats.get('huav_usage_rate', 0)*100:.1f}%")
        print()

    # Per-type comparison
    print("📋 Per-Type Metrics (MRE) Comparison:")
    print("=" * 90)

    huav_type = huav.get('type_metrics', {})
    collab_type = collab.get('type_metrics', {})

    all_types = sorted(set(list(huav_type.keys()) + list(collab_type.keys())))

    print(f"{'Question Type':<15} {'H-UAV Standalone':<25} {'L-UAV+H-UAV':<25} {'Difference':<20}")
    print("-" * 90)

    for qtype in all_types:
        huav_mre = huav_type.get(qtype, {}).get('mre', 0)
        collab_mre = collab_type.get(qtype, {}).get('mre', 0)

        diff = collab_mre - huav_mre
        diff_pct = (diff / huav_mre) * 100 if huav_mre != 0 else 0

        huav_str = f"{huav_mre*100:.2f}%"
        collab_str = f"{collab_mre*100:.2f}%"
        diff_str = f"{diff*100:+.2f} pp ({diff_pct:+.1f}%)"

        print(f"{qtype:<15} {huav_str:<25} {collab_str:<25} {diff_str:<20}")

    print("=" * 90)
    print()

    # Analysis
    print("💡 Analysis:")
    print("-" * 90)

    mae_diff_pct = ((collab_metrics['mae'] - huav_metrics['mae']) / huav_metrics['mae']) * 100
    mre_diff_pp = (collab_metrics['mre'] - huav_metrics['mre']) * 100

    if mae_diff_pct < 0:
        print(f"✅ L-UAV+H-UAV Collaboration OUTPERFORMS H-UAV Standalone:")
        print(f"   - MAE improved by {abs(mae_diff_pct):.2f}%")
        print(f"   - MRE improved by {abs(mre_diff_pp):.2f} percentage points")
        print()
        print("   Interpretation:")
        print("   - L-UAV benefits from H-UAV's LoRA-enhanced memory")
        print("   - Collaboration mechanism is effective")
        print("   - Forced queries every 5 samples help guide L-UAV")
    elif mae_diff_pct > 5:
        print(f"⚠️  H-UAV Standalone OUTPERFORMS L-UAV+H-UAV Collaboration:")
        print(f"   - MAE worse by {mae_diff_pct:.2f}%")
        print(f"   - MRE worse by {mre_diff_pp:.2f} percentage points")
        print()
        print("   Possible reasons:")
        print("   - Memory integration not fully effective")
        print("   - L-UAV may need better integration of memory tokens")
        print("   - Consider adjusting confidence threshold or query frequency")
    else:
        print(f"➡️  Similar Performance:")
        print(f"   - MAE difference: {mae_diff_pct:+.2f}%")
        print(f"   - MRE difference: {mre_diff_pp:+.2f} percentage points")
        print()
        print("   Interpretation:")
        print("   - Both approaches achieve comparable results")
        print("   - L-UAV+H-UAV shows potential but needs refinement")

    print()
    print("Note: Negative difference means L-UAV+H-UAV is better (lower error)")
    print("      Positive difference means H-UAV Standalone is better")
    print("=" * 90)

    # Key findings
    print()
    print("🔑 Key Findings:")
    print("-" * 90)
    print(f"1. H-UAV Standalone (with LoRA MemGen):")
    print(f"   - MAE: {huav_metrics['mae']:.2f}, MRE: {huav_metrics['mre']*100:.2f}%")
    print(f"   - Strong baseline with LoRA-enhanced memory")
    print()
    print(f"2. L-UAV + H-UAV Collaboration:")
    print(f"   - MAE: {collab_metrics['mae']:.2f}, MRE: {collab_metrics['mre']*100:.2f}%")
    if 'collaboration_stats' in collab:
        print(f"   - H-UAV usage rate: {collab_stats.get('huav_usage_rate', 0)*100:.1f}%")
    print(f"   - Demonstrates collaboration mechanism")
    print()
    print(f"3. Comparison:")
    if mae_diff_pct < 0:
        print(f"   ✅ Collaboration improves performance by {abs(mae_diff_pct):.1f}%")
    else:
        print(f"   ⚠️  Standalone currently better by {mae_diff_pct:.1f}%")
    print("=" * 90)

    return True


if __name__ == '__main__':
    success = print_comparison()
    sys.exit(0 if success else 1)
