"""
3-Way Collaboration Comparison Tool V2

Compare:
1. H-UAV Baseline (NO LoRA)
2. H-UAV + MemGen (10-epoch LoRA)
3. L-UAV (3-epoch LoRA) + H-UAV (10-epoch LoRA) Collaboration
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

# Result paths
BASELINE_PATH = "./outputs/comparison_huav_baseline_200/results.json"
HUAV_MEMGEN_PATH = "./outputs/comparison_huav_standalone_200/results.json"
LUAV_HUAV_COLLAB_PATH = "./outputs/comparison_luav_huav_collab_200_v2/results.json"


def load_results(path: str) -> Optional[Dict]:
    """Load results from JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  File not found: {path}")
        return None
    except json.JSONDecodeError:
        print(f"⚠️  Invalid JSON: {path}")
        return None


def print_3way_comparison():
    """Print comprehensive 3-way comparison."""

    print("=" * 80)
    print("3-Way Collaboration Comparison (V2 - LoRA-based)")
    print("=" * 80)
    print()

    # Load all results
    baseline = load_results(BASELINE_PATH)
    huav_memgen = load_results(HUAV_MEMGEN_PATH)
    luav_huav_collab = load_results(LUAV_HUAV_COLLAB_PATH)

    # Check if all results are loaded
    missing = []
    if baseline is None:
        missing.append("Baseline")
    if huav_memgen is None:
        missing.append("H-UAV+MemGen")
    if luav_huav_collab is None:
        missing.append("L-UAV+H-UAV")

    if missing:
        print(f"❌ Missing results: {', '.join(missing)}")
        print()
        print("Please run the experiments first:")
        print("  ./run_collaboration_comparison_v2.sh")
        print()
        return

    # Extract metrics
    baseline_metrics = baseline['metrics']
    huav_metrics = huav_memgen['metrics']
    collab_metrics = luav_huav_collab['metrics']

    # Overall comparison
    print("Overall Metrics Comparison (200 samples)")
    print("-" * 80)
    print(f"{'Metric':<12} | {'Baseline':<15} | {'H-UAV+MemGen':<15} | {'L-UAV+H-UAV':<15} | {'Best':<10}")
    print("-" * 80)

    metrics_to_compare = [
        ('MAE', 'mae', False),
        ('RMSE', 'rmse', False),
        ('MRE', 'mre', True)  # MRE is percentage
    ]

    rankings = {'Baseline': 0, 'H-UAV+MemGen': 0, 'L-UAV+H-UAV': 0}

    for metric_name, metric_key, is_percentage in metrics_to_compare:
        baseline_val = baseline_metrics[metric_key]
        huav_val = huav_metrics[metric_key]
        collab_val = collab_metrics[metric_key]

        # Find best (lowest)
        values = {
            'Baseline': baseline_val,
            'H-UAV+MemGen': huav_val,
            'L-UAV+H-UAV': collab_val
        }
        best_name = min(values, key=values.get)
        rankings[best_name] += 1

        # Format values
        if is_percentage:
            baseline_str = f"{baseline_val:.2%}"
            huav_str = f"{huav_val:.2%}"
            collab_str = f"{collab_val:.2%}"
        else:
            baseline_str = f"{baseline_val:.4f}"
            huav_str = f"{huav_val:.4f}"
            collab_str = f"{collab_val:.4f}"

        print(f"{metric_name:<12} | {baseline_str:<15} | {huav_str:<15} | {collab_str:<15} | {best_name:<10}")

    print("-" * 80)
    print()

    # Gains
    print("Performance Gains (vs Baseline)")
    print("-" * 80)
    print(f"{'Metric':<12} | {'H-UAV+MemGen Gain':<25} | {'L-UAV+H-UAV Gain':<25}")
    print("-" * 80)

    for metric_name, metric_key, is_percentage in metrics_to_compare:
        baseline_val = baseline_metrics[metric_key]
        huav_val = huav_metrics[metric_key]
        collab_val = collab_metrics[metric_key]

        # Calculate gains (reduction in error)
        huav_gain = ((baseline_val - huav_val) / baseline_val) * 100 if baseline_val != 0 else 0
        collab_gain = ((baseline_val - collab_val) / baseline_val) * 100 if baseline_val != 0 else 0

        huav_gain_str = f"{huav_gain:+.2f}%" if huav_gain != 0 else "0.00%"
        collab_gain_str = f"{collab_gain:+.2f}%" if collab_gain != 0 else "0.00%"

        print(f"{metric_name:<12} | {huav_gain_str:<25} | {collab_gain_str:<25}")

    print("-" * 80)
    print()

    # Collaboration statistics
    if 'collaboration_stats' in luav_huav_collab:
        collab_stats = luav_huav_collab['collaboration_stats']
        print("Collaboration Statistics (L-UAV + H-UAV)")
        print("-" * 80)
        print(f"  L-UAV LoRA: {collab_stats.get('luav_lora_epochs', 3)} epochs (basic memory)")
        print(f"  H-UAV LoRA: {collab_stats.get('huav_lora_epochs', 10)} epochs (enhanced memory)")
        print(f"  H-UAV requests: {collab_stats.get('huav_requests', 0)}")
        print(f"  Standalone inferences: {collab_stats.get('standalone_inferences', 0)}")
        print(f"  H-UAV usage rate: {collab_stats.get('huav_usage_rate', 0):.2%}")
        print()

    # Per-type comparison
    print("Per-Type Performance")
    print("-" * 80)

    # Get all question types
    all_types = set()
    if 'type_metrics' in baseline:
        all_types.update(baseline['type_metrics'].keys())
    if 'type_metrics' in huav_memgen:
        all_types.update(huav_memgen['type_metrics'].keys())
    if 'type_metrics' in luav_huav_collab:
        all_types.update(luav_huav_collab['type_metrics'].keys())

    for qtype in sorted(all_types):
        print(f"\n{qtype.upper()}:")
        print(f"  {'Metric':<10} | {'Baseline':<12} | {'H-UAV+MemGen':<12} | {'L-UAV+H-UAV':<12}")
        print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

        baseline_type = baseline.get('type_metrics', {}).get(qtype, {})
        huav_type = huav_memgen.get('type_metrics', {}).get(qtype, {})
        collab_type = luav_huav_collab.get('type_metrics', {}).get(qtype, {})

        for metric in ['mae', 'rmse', 'mre']:
            baseline_val = baseline_type.get(metric, 0)
            huav_val = huav_type.get(metric, 0)
            collab_val = collab_type.get(metric, 0)

            if metric == 'mre':
                baseline_str = f"{baseline_val:.2%}"
                huav_str = f"{huav_val:.2%}"
                collab_str = f"{collab_val:.2%}"
            else:
                baseline_str = f"{baseline_val:.2f}"
                huav_str = f"{huav_val:.2f}"
                collab_str = f"{collab_val:.2f}"

            print(f"  {metric.upper():<10} | {baseline_str:<12} | {huav_str:<12} | {collab_str:<12}")

    print()
    print("=" * 80)

    # Overall ranking
    print("Overall Ranking")
    print("=" * 80)

    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    medals = ['🥇', '🥈', '🥉']

    for i, (name, wins) in enumerate(sorted_rankings):
        medal = medals[i] if i < len(medals) else '  '
        print(f"  {medal} {name}: {wins}/3 best metrics")

    print()

    # Analysis
    print("Analysis")
    print("=" * 80)

    # Check if LoRA helps
    huav_mae_gain = ((baseline_metrics['mae'] - huav_metrics['mae']) / baseline_metrics['mae']) * 100
    if huav_mae_gain > 0:
        print(f"  ✓ LoRA MemGen (10 epochs) improves MAE by {huav_mae_gain:.2f}%")
    else:
        print(f"  ✗ LoRA MemGen shows no improvement (MAE change: {huav_mae_gain:.2f}%)")

    # Check if collaboration helps
    collab_mae_gain = ((baseline_metrics['mae'] - collab_metrics['mae']) / baseline_metrics['mae']) * 100
    if collab_mae_gain > 0:
        print(f"  ✓ L-UAV+H-UAV collaboration improves MAE by {collab_mae_gain:.2f}%")
    else:
        print(f"  ✗ L-UAV+H-UAV collaboration shows no improvement (MAE change: {collab_mae_gain:.2f}%)")

    # Compare H-UAV standalone vs collaboration
    huav_vs_collab = ((huav_metrics['mae'] - collab_metrics['mae']) / huav_metrics['mae']) * 100
    if abs(huav_vs_collab) < 1:
        print(f"  ≈ H-UAV standalone and L-UAV+H-UAV show similar performance (diff: {huav_vs_collab:.2f}%)")
    elif huav_vs_collab > 0:
        print(f"  ✓ L-UAV+H-UAV outperforms H-UAV standalone by {huav_vs_collab:.2f}%")
    else:
        print(f"  ✓ H-UAV standalone outperforms L-UAV+H-UAV by {-huav_vs_collab:.2f}%")

    print()

    # Recommendations
    print("Recommendations")
    print("=" * 80)

    best_approach = sorted_rankings[0][0]
    print(f"  Best approach: {best_approach}")
    print()

    if best_approach == "H-UAV+MemGen":
        print("  → Use H-UAV with 10-epoch LoRA for best performance")
        print("  → LoRA memory enhancement is effective")
    elif best_approach == "L-UAV+H-UAV":
        print("  → Use L-UAV+H-UAV collaboration for best performance")
        print("  → Hierarchical architecture with memory injection is effective")
    else:
        print("  ⚠️  Baseline performs best - LoRA and collaboration need improvement")
        print("  → Consider: longer training, better hyperparameters, or architecture changes")

    print()
    print("=" * 80)


def main():
    print_3way_comparison()


if __name__ == "__main__":
    main()
