#!/usr/bin/env python3
"""
Compare L-UAV results with and without trained MAC memory.

This script compares two experimental conditions:
- Experiment A: L-UAV + Untrained MAC (random initialization)
- Experiment B: L-UAV + Trained MAC (after test-time learning)

Usage:
    python compare_memory_experiments.py \
        --untrained ./outputs/hierarchical_uav/task1_l_uav_UNTRAINED.jsonl \
        --trained ./outputs/hierarchical_uav/task1_l_uav_TRAINED.jsonl \
        --ground-truth /mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import re


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    if not Path(file_path).exists():
        print(f"Warning: File not found: {file_path}")
        return []

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    answer = ' '.join(answer.split())
    answer = answer.lower()
    answer = answer.rstrip('.,!?;')
    return answer


def extract_ground_truth(sample: Dict) -> str:
    """Extract ground truth answer from sample."""
    # Try different possible keys
    if 'answer' in sample:
        return str(sample['answer'])
    elif 'gt' in sample:
        return str(sample['gt'])
    elif 'ground_truth' in sample:
        return str(sample['ground_truth'])
    elif 'conversations' in sample:
        # LLaVA format
        for conv in sample['conversations']:
            if conv.get('from') == 'gpt':
                return conv.get('value', '')
    return ""


def calculate_accuracy(results: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Calculate accuracy metrics."""

    # Create GT lookup by image_id
    gt_lookup = {}
    for gt in ground_truth:
        image_id = gt.get('image', gt.get('image_id', ''))
        gt_answer = extract_ground_truth(gt)
        if image_id and gt_answer:
            gt_lookup[image_id] = normalize_answer(gt_answer)

    total = 0
    correct = 0
    memory_used = 0
    memory_correct = 0
    local_only = 0
    local_correct = 0

    answers_diversity = set()

    for result in results:
        image_id = result.get('image', result.get('image_id', ''))
        pred_answer = normalize_answer(result.get('answer', ''))
        used_memory = result.get('used_huav_memory', False)

        if not image_id or not pred_answer:
            continue

        total += 1
        answers_diversity.add(pred_answer)

        # Check if correct
        if image_id in gt_lookup:
            is_correct = (pred_answer == gt_lookup[image_id])

            if is_correct:
                correct += 1

            # Track memory usage statistics
            if used_memory:
                memory_used += 1
                if is_correct:
                    memory_correct += 1
            else:
                local_only += 1
                if is_correct:
                    local_correct += 1

    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0.0,
        'memory_used': memory_used,
        'memory_correct': memory_correct,
        'memory_accuracy': memory_correct / memory_used if memory_used > 0 else 0.0,
        'local_only': local_only,
        'local_correct': local_correct,
        'local_accuracy': local_correct / local_only if local_only > 0 else 0.0,
        'answer_diversity': len(answers_diversity),
        'diversity_rate': len(answers_diversity) / total if total > 0 else 0.0
    }


def print_comparison(untrained_metrics: Dict, trained_metrics: Dict):
    """Print comparison table."""

    print("\n" + "="*80)
    print("MEMORY EFFECTIVENESS COMPARISON")
    print("="*80)

    print("\n📊 Overall Performance:")
    print("-" * 80)
    print(f"{'Metric':<30} {'Untrained MAC':<20} {'Trained MAC':<20} {'Improvement':<10}")
    print("-" * 80)

    # Overall accuracy
    untrained_acc = untrained_metrics['accuracy'] * 100
    trained_acc = trained_metrics['accuracy'] * 100
    improvement = trained_acc - untrained_acc

    print(f"{'Total Samples':<30} {untrained_metrics['total']:<20} {trained_metrics['total']:<20} {'-':<10}")
    print(f"{'Overall Accuracy':<30} {untrained_acc:<19.2f}% {trained_acc:<19.2f}% {improvement:+.2f}%")

    # Answer diversity
    untrained_div = untrained_metrics['diversity_rate'] * 100
    trained_div = trained_metrics['diversity_rate'] * 100
    div_improvement = trained_div - untrained_div

    print(f"{'Unique Answers':<30} {untrained_metrics['answer_diversity']:<20} {trained_metrics['answer_diversity']:<20} {'-':<10}")
    print(f"{'Answer Diversity':<30} {untrained_div:<19.2f}% {trained_div:<19.2f}% {div_improvement:+.2f}%")

    print("\n🔍 Memory Usage Statistics:")
    print("-" * 80)
    print(f"{'Metric':<30} {'Untrained MAC':<20} {'Trained MAC':<20} {'Improvement':<10}")
    print("-" * 80)

    # Memory usage
    print(f"{'Samples Using Memory':<30} {untrained_metrics['memory_used']:<20} {trained_metrics['memory_used']:<20} {'-':<10}")

    untrained_mem_acc = untrained_metrics['memory_accuracy'] * 100
    trained_mem_acc = trained_metrics['memory_accuracy'] * 100
    mem_improvement = trained_mem_acc - untrained_mem_acc

    print(f"{'Memory-Augmented Accuracy':<30} {untrained_mem_acc:<19.2f}% {trained_mem_acc:<19.2f}% {mem_improvement:+.2f}%")

    # Local inference
    print(f"{'Samples Using Local Only':<30} {untrained_metrics['local_only']:<20} {trained_metrics['local_only']:<20} {'-':<10}")

    untrained_local_acc = untrained_metrics['local_accuracy'] * 100
    trained_local_acc = trained_metrics['local_accuracy'] * 100
    local_improvement = trained_local_acc - untrained_local_acc

    print(f"{'Local-Only Accuracy':<30} {untrained_local_acc:<19.2f}% {trained_local_acc:<19.2f}% {local_improvement:+.2f}%")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if improvement > 0:
        print(f"✅ Training MAC memory IMPROVED accuracy by {improvement:.2f}%")
    elif improvement < 0:
        print(f"❌ Training MAC memory DECREASED accuracy by {abs(improvement):.2f}%")
    else:
        print(f"➖ Training MAC memory had NO EFFECT on accuracy")

    if mem_improvement > 0:
        print(f"✅ Memory-augmented samples improved by {mem_improvement:.2f}%")
    elif mem_improvement < 0:
        print(f"❌ Memory-augmented samples decreased by {abs(mem_improvement):.2f}%")

    if div_improvement > 0:
        print(f"✅ Answer diversity increased by {div_improvement:.2f}%")
        print(f"   (Trained MAC produces more varied answers, less random noise)")

    print("="*80 + "\n")


def save_detailed_report(untrained_metrics: Dict, trained_metrics: Dict, output_path: str):
    """Save detailed JSON report."""
    report = {
        'untrained_mac': untrained_metrics,
        'trained_mac': trained_metrics,
        'improvements': {
            'overall_accuracy': (trained_metrics['accuracy'] - untrained_metrics['accuracy']) * 100,
            'memory_accuracy': (trained_metrics['memory_accuracy'] - untrained_metrics['memory_accuracy']) * 100,
            'local_accuracy': (trained_metrics['local_accuracy'] - untrained_metrics['local_accuracy']) * 100,
            'answer_diversity': (trained_metrics['diversity_rate'] - untrained_metrics['diversity_rate']) * 100
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Detailed report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare L-UAV experiments with/without trained MAC")

    parser.add_argument('--untrained', required=True,
                       help='L-UAV results with untrained MAC (Experiment A)')
    parser.add_argument('--trained', required=True,
                       help='L-UAV results with trained MAC (Experiment B)')
    parser.add_argument('--ground-truth', required=True,
                       help='Ground truth JSONL file')
    parser.add_argument('--output-report', default='./memory_comparison_report.json',
                       help='Output detailed report JSON file')

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    untrained_results = load_jsonl(args.untrained)
    trained_results = load_jsonl(args.trained)
    ground_truth = load_jsonl(args.ground_truth)

    print(f"  Untrained MAC results: {len(untrained_results)} samples")
    print(f"  Trained MAC results: {len(trained_results)} samples")
    print(f"  Ground truth: {len(ground_truth)} samples")

    # Calculate metrics
    print("\nCalculating metrics...")
    untrained_metrics = calculate_accuracy(untrained_results, ground_truth)
    trained_metrics = calculate_accuracy(trained_results, ground_truth)

    # Print comparison
    print_comparison(untrained_metrics, trained_metrics)

    # Save detailed report
    save_detailed_report(untrained_metrics, trained_metrics, args.output_report)


if __name__ == '__main__':
    main()
