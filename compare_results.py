#!/usr/bin/env python3
"""
Compare evaluation results from different UAV configurations.

Compares:
1. L-UAV standalone (baseline)
2. H-UAV standalone (MemGen upper bound)
3. L-UAV + H-UAV collaboration (actual deployment)

Usage:
    python compare_results.py
"""

import json
import os
from typing import Dict, Optional


def load_results(path: str) -> Optional[Dict]:
    """Load results from JSON file."""
    if not os.path.exists(path):
        return None

    with open(path) as f:
        return json.load(f)


def print_metrics(name: str, results: Optional[Dict], indent: str = ""):
    """Print metrics for a configuration."""
    if results is None:
        print(f"{indent}{name}: No results found")
        return

    metrics = results.get('metrics', {})
    mae = metrics.get('mae', 0)
    rmse = metrics.get('rmse', 0)
    mre = metrics.get('mre', 0)

    print(f"{indent}{name}:")
    print(f"{indent}  MAE:  {mae:8.2f}")
    print(f"{indent}  RMSE: {rmse:8.2f}")
    print(f"{indent}  MRE:  {mre:7.2%}")

    # Additional stats for collaborative mode
    if 'stats' in results:
        stats = results['stats']
        total = stats.get('total_queries', 0)
        huav_reqs = stats.get('huav_requests', 0)
        if total > 0:
            usage_rate = huav_reqs / total * 100
            print(f"{indent}  H-UAV Usage: {huav_reqs}/{total} ({usage_rate:.1f}%)")


def compare_results():
    """Compare all evaluation results."""
    print("=" * 80)
    print("Hierarchical UAV System - Performance Comparison")
    print("=" * 80)
    print()

    # Paths
    luav_standalone = "./outputs/eval_luav_standalone/results.json"
    huav_standalone = "./outputs/eval_huav_standalone/results.json"
    collaborative = "./outputs/eval_luav_with_huav/results.json"

    # Load results
    luav_results = load_results(luav_standalone)
    huav_results = load_results(huav_standalone)
    collab_results = load_results(collaborative)

    # Print individual results
    print_metrics("1. L-UAV Standalone (Baseline)", luav_results)
    print()
    print_metrics("2. H-UAV Standalone (MemGen Upper Bound)", huav_results)
    print()
    print_metrics("3. L-UAV + H-UAV Collaboration (Actual Deployment)", collab_results)
    print()

    # Calculate improvements
    print("=" * 80)
    print("Performance Improvements")
    print("=" * 80)
    print()

    if luav_results and huav_results:
        luav_mae = luav_results['metrics']['mae']
        huav_mae = huav_results['metrics']['mae']
        improvement = (luav_mae - huav_mae) / luav_mae * 100
        print(f"H-UAV vs L-UAV Standalone:")
        print(f"  MAE improvement: {improvement:.1f}% ({luav_mae:.2f} → {huav_mae:.2f})")
        print()

    if luav_results and collab_results:
        luav_mae = luav_results['metrics']['mae']
        collab_mae = collab_results['metrics']['mae']
        improvement = (luav_mae - collab_mae) / luav_mae * 100
        print(f"Collaborative vs L-UAV Standalone:")
        print(f"  MAE improvement: {improvement:.1f}% ({luav_mae:.2f} → {collab_mae:.2f})")
        print()

    if huav_results and collab_results:
        huav_mae = huav_results['metrics']['mae']
        collab_mae = collab_results['metrics']['mae']
        gap = (collab_mae - huav_mae) / huav_mae * 100
        print(f"Collaborative vs H-UAV Standalone:")
        print(f"  Performance gap: {gap:.1f}% ({collab_mae:.2f} vs {huav_mae:.2f})")
        print(f"  (Gap shows room for improvement in collaboration mechanism)")
        print()

    # Per-type comparison for H-UAV standalone
    if huav_results and 'type_metrics' in huav_results:
        print("=" * 80)
        print("H-UAV Per-Type Performance")
        print("=" * 80)
        print()

        type_metrics = huav_results['type_metrics']
        print(f"{'Type':<12} {'Count':>6} {'MAE':>10} {'RMSE':>10} {'MRE':>10}")
        print("-" * 80)

        for qtype, metrics in sorted(type_metrics.items()):
            count = metrics['count']
            mae = metrics['mae']
            rmse = metrics['rmse']
            mre = metrics['mre']
            print(f"{qtype:<12} {count:>6} {mae:>10.2f} {rmse:>10.2f} {mre:>9.2%}")

        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()

    if luav_results and huav_results and collab_results:
        print("✓ All three evaluations completed")
        print()
        print("Key Findings:")
        print("  - H-UAV standalone shows LoRA memory weaver effectiveness")
        print("  - Collaborative mode balances performance and efficiency")
        print("  - H-UAV usage rate indicates self-matching quality")
        print()

        # Check if results make sense
        luav_mae = luav_results['metrics']['mae']
        huav_mae = huav_results['metrics']['mae']
        collab_mae = collab_results['metrics']['mae']

        if huav_mae < luav_mae:
            print("✓ H-UAV outperforms L-UAV (LoRA memory is effective)")
        else:
            print("⚠️  H-UAV should outperform L-UAV - check training")

        if collab_mae < luav_mae:
            print("✓ Collaboration improves over baseline")
        else:
            print("⚠️  Collaboration should improve over baseline")

        if huav_mae <= collab_mae <= luav_mae:
            print("✓ Collaborative performance is between H-UAV and L-UAV (expected)")
        else:
            print("⚠️  Collaborative performance should be between H-UAV and L-UAV")

        print()
    else:
        missing = []
        if not luav_results:
            missing.append("L-UAV standalone")
        if not huav_results:
            missing.append("H-UAV standalone")
        if not collab_results:
            missing.append("Collaborative mode")

        print(f"⚠️  Missing results: {', '.join(missing)}")
        print()
        print("Run the following to complete evaluation:")
        if not luav_results:
            print("  ./start_luav_standalone.sh")
        if not huav_results:
            print("  ./start_huav_standalone.sh")
        if not collab_results:
            print("  # Terminal 1: ./start_huav_lora.sh")
            print("  # Terminal 2: ./start_luav_with_huav.sh")
        print()


if __name__ == '__main__':
    compare_results()
