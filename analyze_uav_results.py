#!/usr/bin/env python3
"""
Analyze H-UAV and L-UAV evaluation results.

Compares:
1. L-UAV local inference (no H-UAV memory)
2. L-UAV with H-UAV memory (after training)
3. H-UAV direct inference

Usage:
    python analyze_uav_results.py \
        --luav-results ./outputs/hierarchical_uav/task1_l-uav_results.jsonl \
        --huav-results ./outputs/hierarchical_uav/task1_h-uav_results.jsonl \
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
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    # Lowercase
    answer = answer.lower()
    # Remove punctuation at end
    answer = answer.rstrip('.,!?;')
    return answer


def extract_bbox_answer(answer: str) -> str:
    """Extract bbox coordinates if present."""
    # Look for bbox pattern like [x, y, w, h]
    bbox_match = re.search(r'\[[\d\s,\.]+\]', answer)
    if bbox_match:
        return bbox_match.group(0)
    return answer


def extract_ground_truth_answer(gt_item: Dict) -> str:
    """Extract ground truth answer from various possible formats."""
    # Try gt field (AirSpatial format)
    if 'gt' in gt_item:
        return gt_item['gt']

    # Try conversations format (LLaVA style)
    conversations = gt_item.get('conversations', [])
    if len(conversations) >= 2:
        return conversations[1].get('value', '')

    # Try direct answer field
    if 'answer' in gt_item:
        return gt_item['answer']

    # Try text field
    if 'text' in gt_item:
        return gt_item['text']

    # Try output field
    if 'output' in gt_item:
        return gt_item['output']

    return ''


def calculate_accuracy(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Calculate accuracy metrics."""

    # Create ground truth mapping
    gt_map = {}
    for gt in ground_truth:
        question_id = gt.get('question_id', gt.get('id'))
        if question_id:
            gt_map[question_id] = gt

    total = 0
    correct = 0
    partial_correct = 0
    answer_distribution = defaultdict(int)
    error_analysis = {
        'no_match': [],
        'partial_match': [],
        'correct_match': []
    }

    for pred in predictions:
        question_id = pred.get('question_id', pred.get('id'))
        if not question_id or question_id not in gt_map:
            continue

        total += 1
        pred_answer = normalize_answer(pred.get('text', pred.get('answer', '')))
        gt_answer = normalize_answer(extract_ground_truth_answer(gt_map[question_id]))

        answer_distribution[pred_answer[:50]] += 1  # Track first 50 chars

        # Check for exact match
        if pred_answer == gt_answer:
            correct += 1
            error_analysis['correct_match'].append({
                'question_id': question_id,
                'question': pred.get('prompt', ''),
                'pred': pred_answer,
                'gt': gt_answer
            })
        # Check for partial match (answer contains ground truth or vice versa)
        elif gt_answer in pred_answer or pred_answer in gt_answer:
            partial_correct += 1
            error_analysis['partial_match'].append({
                'question_id': question_id,
                'question': pred.get('prompt', ''),
                'pred': pred_answer,
                'gt': gt_answer
            })
        else:
            error_analysis['no_match'].append({
                'question_id': question_id,
                'question': pred.get('prompt', ''),
                'pred': pred_answer,
                'gt': gt_answer
            })

    return {
        'total': total,
        'correct': correct,
        'partial_correct': partial_correct,
        'accuracy': correct / total if total > 0 else 0.0,
        'partial_accuracy': (correct + partial_correct) / total if total > 0 else 0.0,
        'unique_answers': len(answer_distribution),
        'answer_distribution': dict(answer_distribution),
        'error_analysis': error_analysis
    }


def analyze_memory_usage(luav_results: List[Dict]) -> Dict:
    """Analyze memory query statistics from L-UAV results."""
    stats = {
        'total_samples': 0,
        'used_memory': 0,
        'used_local': 0,
        'avg_self_matching_score': 0.0,
        'memory_triggered_samples': []
    }

    self_matching_scores = []

    for result in luav_results:
        stats['total_samples'] += 1

        # Check if memory was used
        if result.get('used_memory', False):
            stats['used_memory'] += 1
            stats['memory_triggered_samples'].append({
                'question_id': result.get('question_id'),
                'self_matching': result.get('self_matching_score', 0.0),
                'answer': result.get('text', '')[:100]
            })
        else:
            stats['used_local'] += 1

        # Track self-matching scores
        if 'self_matching_score' in result:
            self_matching_scores.append(result['self_matching_score'])

    if self_matching_scores:
        stats['avg_self_matching_score'] = sum(self_matching_scores) / len(self_matching_scores)

    stats['memory_usage_rate'] = stats['used_memory'] / stats['total_samples'] if stats['total_samples'] > 0 else 0.0

    return stats


def print_comparison(luav_metrics: Dict, huav_metrics: Dict = None, memory_stats: Dict = None):
    """Print comparison report."""

    print("\n" + "=" * 80)
    print("UAV EVALUATION RESULTS - TRAINED MAC LAYERS")
    print("=" * 80)

    print("\n📊 OVERALL PERFORMANCE:")
    print("-" * 80)
    if huav_metrics:
        print(f"{'Metric':<30} {'L-UAV':<20} {'H-UAV':<20}")
        print("-" * 80)
        print(f"{'Total Samples':<30} {luav_metrics['total']:<20} {huav_metrics['total']:<20}")
        print(f"{'Exact Match Accuracy':<30} {luav_metrics['accuracy']:<20.2%} {huav_metrics['accuracy']:<20.2%}")
        print(f"{'Partial Match Accuracy':<30} {luav_metrics['partial_accuracy']:<20.2%} {huav_metrics['partial_accuracy']:<20.2%}")
        print(f"{'Unique Answers':<30} {luav_metrics['unique_answers']:<20} {huav_metrics['unique_answers']:<20}")
    else:
        print(f"{'Metric':<30} {'L-UAV':<20}")
        print("-" * 80)
        print(f"{'Total Samples':<30} {luav_metrics['total']:<20}")
        print(f"{'Exact Match Accuracy':<30} {luav_metrics['accuracy']:<20.2%}")
        print(f"{'Partial Match Accuracy':<30} {luav_metrics['partial_accuracy']:<20.2%}")
        print(f"{'Unique Answers':<30} {luav_metrics['unique_answers']:<20}")

    if memory_stats:
        print("\n🧠 MEMORY USAGE ANALYSIS (L-UAV):")
        print("-" * 80)
        print(f"  Total samples: {memory_stats['total_samples']}")
        print(f"  Used H-UAV memory: {memory_stats['used_memory']} ({memory_stats['memory_usage_rate']:.1%})")
        print(f"  Used local KB: {memory_stats['used_local']} ({(1-memory_stats['memory_usage_rate']):.1%})")
        print(f"  Avg self-matching score: {memory_stats['avg_self_matching_score']:.3f}")

    print("\n📈 ACCURACY BREAKDOWN:")
    print("-" * 80)
    print(f"L-UAV:")
    print(f"  ✓ Correct: {luav_metrics['correct']} ({luav_metrics['accuracy']:.2%})")
    print(f"  ≈ Partial: {luav_metrics['partial_correct']} ({luav_metrics['partial_correct']/luav_metrics['total']:.2%})")
    print(f"  ✗ Wrong: {luav_metrics['total'] - luav_metrics['correct'] - luav_metrics['partial_correct']}")

    if huav_metrics:
        print(f"\nH-UAV:")
        print(f"  ✓ Correct: {huav_metrics['correct']} ({huav_metrics['accuracy']:.2%})")
        print(f"  ≈ Partial: {huav_metrics['partial_correct']} ({huav_metrics['partial_correct']/huav_metrics['total']:.2%})")
        print(f"  ✗ Wrong: {huav_metrics['total'] - huav_metrics['correct'] - huav_metrics['partial_correct']}")

    print("\n🎯 KEY INSIGHTS:")
    print("-" * 80)

    # Compare with baseline (from previous session: 3.1% accuracy)
    baseline_accuracy = 0.031
    improvement = (luav_metrics['accuracy'] - baseline_accuracy) / baseline_accuracy if baseline_accuracy > 0 else 0

    print(f"  Previous L-UAV accuracy (untrained MAC): ~{baseline_accuracy:.1%}")
    print(f"  Current L-UAV accuracy (trained MAC): {luav_metrics['accuracy']:.2%}")
    print(f"  Improvement: {improvement:+.1%} ({improvement*100:.0f}x better)" if improvement > 0 else f"  Change: {improvement:.1%}")

    if memory_stats and memory_stats['used_memory'] > 0:
        memory_samples = [s for s in luav_metrics['error_analysis']['correct_match']
                         if any(ms['question_id'] == s['question_id']
                               for ms in memory_stats['memory_triggered_samples'])]
        memory_accuracy = len(memory_samples) / memory_stats['used_memory'] if memory_stats['used_memory'] > 0 else 0
        print(f"\n  Accuracy on memory-augmented samples: {memory_accuracy:.2%}")
        print(f"  (Trained MAC helps L-UAV answer {len(memory_samples)}/{memory_stats['used_memory']} memory queries correctly)")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze UAV evaluation results')
    parser.add_argument('--luav-results', type=str, required=True, help='L-UAV results JSONL file')
    parser.add_argument('--huav-results', type=str, help='H-UAV results JSONL file (optional)')
    parser.add_argument('--ground-truth', type=str, required=True, help='Ground truth JSONL file')
    parser.add_argument('--output-report', type=str, help='Output report JSON file')

    args = parser.parse_args()

    # Load data
    print("Loading results...")
    luav_results = load_jsonl(args.luav_results)
    ground_truth = load_jsonl(args.ground_truth)

    print(f"  L-UAV: {len(luav_results)} results")

    # Load H-UAV results if provided
    huav_results = None
    huav_metrics = None
    if args.huav_results:
        huav_results = load_jsonl(args.huav_results)
        print(f"  H-UAV: {len(huav_results)} results")

    print(f"  Ground truth: {len(ground_truth)} samples")

    # Calculate metrics
    print("\nCalculating metrics...")
    luav_metrics = calculate_accuracy(luav_results, ground_truth)
    if huav_results:
        huav_metrics = calculate_accuracy(huav_results, ground_truth)
    memory_stats = analyze_memory_usage(luav_results)

    # Print comparison
    print_comparison(luav_metrics, huav_metrics, memory_stats)

    # Save detailed report
    if args.output_report:
        report = {
            'luav_metrics': luav_metrics,
            'huav_metrics': huav_metrics,
            'memory_stats': memory_stats,
            'timestamp': str(Path(args.luav_results).stat().st_mtime)
        }

        with open(args.output_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Detailed report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
