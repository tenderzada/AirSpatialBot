#!/usr/bin/env python3
"""
Compare L-UAV baseline (standalone) vs memory-augmented performance.

Usage:
    python compare_baseline_vs_memory.py <baseline.jsonl> <memory_augmented.jsonl>
"""

import sys
import json
from collections import defaultdict
from typing import Dict, List


def load_results(filepath: str) -> List[Dict]:
    """Load JSONL results file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    return str(text).lower().strip()


def evaluate_accuracy(results: List[Dict]) -> Dict:
    """Evaluate accuracy and source distribution."""
    stats = {
        'total': len(results),
        'correct': 0,
        'accuracy': 0.0,
        'source_dist': defaultdict(int),
        'qtype_accuracy': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'memory_used': 0,
        'kb_used': 0,
        'local_only': 0
    }

    for result in results:
        answer = normalize_answer(result.get('answer', ''))
        gt = normalize_answer(result.get('ground_truth', ''))
        source = result.get('source', 'unknown')
        qtype = result.get('qtype', 'unknown')

        # Check correctness
        is_correct = (answer == gt) or (gt in answer) or (answer in gt)
        if is_correct:
            stats['correct'] += 1
            stats['qtype_accuracy'][qtype]['correct'] += 1

        stats['qtype_accuracy'][qtype]['total'] += 1
        stats['source_dist'][source] += 1

        # Track source types
        if source == 'huav':
            stats['memory_used'] += 1
        elif source == 'knowledge_base':
            stats['kb_used'] += 1
        elif source in ['local', 'local_fallback']:
            stats['local_only'] += 1

    stats['accuracy'] = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0.0

    # Calculate per-qtype accuracy
    for qtype in stats['qtype_accuracy']:
        qstats = stats['qtype_accuracy'][qtype]
        qstats['accuracy'] = qstats['correct'] / qstats['total'] * 100 if qstats['total'] > 0 else 0.0

    return stats


def print_comparison(baseline_stats: Dict, memory_stats: Dict, baseline_name: str, memory_name: str):
    """Print detailed comparison."""
    print("=" * 80)
    print(f"L-UAV Performance Comparison: Baseline vs Memory-Augmented")
    print("=" * 80)
    print()

    # Overall accuracy
    print("Overall Performance:")
    print(f"  {baseline_name:30s}: {baseline_stats['correct']:4d}/{baseline_stats['total']:4d} = {baseline_stats['accuracy']:6.2f}%")
    print(f"  {memory_name:30s}: {memory_stats['correct']:4d}/{memory_stats['total']:4d} = {memory_stats['accuracy']:6.2f}%")

    acc_diff = memory_stats['accuracy'] - baseline_stats['accuracy']
    print(f"  {'Improvement':30s}: {acc_diff:+6.2f}%")
    print()

    # Source distribution
    print("Source Distribution:")
    print(f"  {'Source':<20s} {'Baseline':<15s} {'Memory-Augmented':<15s}")
    print(f"  {'-'*20} {'-'*15} {'-'*15}")

    all_sources = set(baseline_stats['source_dist'].keys()) | set(memory_stats['source_dist'].keys())
    for source in sorted(all_sources):
        b_count = baseline_stats['source_dist'].get(source, 0)
        m_count = memory_stats['source_dist'].get(source, 0)
        b_pct = b_count / baseline_stats['total'] * 100 if baseline_stats['total'] > 0 else 0
        m_pct = m_count / memory_stats['total'] * 100 if memory_stats['total'] > 0 else 0
        print(f"  {source:<20s} {b_count:4d} ({b_pct:5.1f}%)  {m_count:4d} ({m_pct:5.1f}%)")
    print()

    # Memory usage stats
    print("Memory System Usage:")
    print(f"  Baseline H-UAV queries : {baseline_stats['memory_used']:4d} ({baseline_stats['memory_used']/baseline_stats['total']*100:5.1f}%)")
    print(f"  Memory-Aug H-UAV queries: {memory_stats['memory_used']:4d} ({memory_stats['memory_used']/memory_stats['total']*100:5.1f}%)")
    print()

    # Per-question-type accuracy
    print("Per Question Type Accuracy:")
    print(f"  {'Question Type':<15s} {'Baseline':<20s} {'Memory-Augmented':<20s} {'Improvement':<10s}")
    print(f"  {'-'*15} {'-'*20} {'-'*20} {'-'*10}")

    all_qtypes = set(baseline_stats['qtype_accuracy'].keys()) | set(memory_stats['qtype_accuracy'].keys())
    for qtype in sorted(all_qtypes):
        b_qstats = baseline_stats['qtype_accuracy'].get(qtype, {'correct': 0, 'total': 0, 'accuracy': 0.0})
        m_qstats = memory_stats['qtype_accuracy'].get(qtype, {'correct': 0, 'total': 0, 'accuracy': 0.0})

        b_str = f"{b_qstats['correct']}/{b_qstats['total']} ({b_qstats['accuracy']:.1f}%)"
        m_str = f"{m_qstats['correct']}/{m_qstats['total']} ({m_qstats['accuracy']:.1f}%)"
        diff = m_qstats['accuracy'] - b_qstats['accuracy']

        print(f"  {qtype:<15s} {b_str:<20s} {m_str:<20s} {diff:+6.2f}%")
    print()

    # Summary
    print("=" * 80)
    print("Summary:")
    if acc_diff > 0:
        print(f"  ✓ Memory augmentation improved accuracy by {acc_diff:.2f}%")
    elif acc_diff < 0:
        print(f"  ✗ Memory augmentation decreased accuracy by {acc_diff:.2f}%")
    else:
        print(f"  → No change in accuracy")

    memory_pct = memory_stats['memory_used'] / memory_stats['total'] * 100 if memory_stats['total'] > 0 else 0
    print(f"  → {memory_stats['memory_used']} samples ({memory_pct:.1f}%) used H-UAV memory")
    print("=" * 80)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_baseline_vs_memory.py <baseline.jsonl> <memory_augmented.jsonl>")
        print()
        print("Example:")
        print("  python compare_baseline_vs_memory.py \\")
        print("    outputs/hierarchical_uav/task1_l-uav_standalone.jsonl \\")
        print("    outputs/hierarchical_uav/task1_l-uav_results.jsonl")
        sys.exit(1)

    baseline_file = sys.argv[1]
    memory_file = sys.argv[2]

    print(f"Loading baseline results from: {baseline_file}")
    baseline_results = load_results(baseline_file)

    print(f"Loading memory-augmented results from: {memory_file}")
    memory_results = load_results(memory_file)
    print()

    baseline_stats = evaluate_accuracy(baseline_results)
    memory_stats = evaluate_accuracy(memory_results)

    print_comparison(
        baseline_stats,
        memory_stats,
        "Baseline (No Memory)",
        "Memory-Augmented"
    )


if __name__ == '__main__':
    main()
