"""
Analyze H-UAV Memory Quality

This script analyzes:
1. H-UAV's own prediction accuracy
2. Self-matching score distribution
3. Correlation between self-matching scores and accuracy
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def extract_number(answer_text):
    """Extract numeric answer from text."""
    import re

    # Handle direct numbers
    if isinstance(answer_text, (int, float)):
        return float(answer_text)

    # Extract from text like "This car has 5 seats."
    match = re.search(r'\b(\d+)\b', str(answer_text))
    if match:
        return float(match.group(1))

    return None

def analyze_huav_results(results_file):
    """Analyze H-UAV memory quality from L-UAV results."""

    print("=" * 70)
    print("H-UAV Memory Quality Analysis")
    print("=" * 70)

    # Load L-UAV results
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"\nTotal samples: {len(results)}")

    # Separate by source
    huav_samples = [r for r in results if r.get('cache_hit', False) or r.get('source') == 'huav']
    local_samples = [r for r in results if not r.get('cache_hit', False) and r.get('source') != 'huav']

    print(f"H-UAV (cache hit) samples: {len(huav_samples)}")
    print(f"Local samples: {len(local_samples)}")

    # Analyze H-UAV samples
    if huav_samples:
        print("\n" + "=" * 70)
        print("H-UAV Memory Analysis")
        print("=" * 70)

        # Accuracy analysis
        correct = 0
        total = 0
        score_ranges = defaultdict(lambda: {'correct': 0, 'total': 0})
        all_scores = []

        for sample in huav_samples:
            pred = extract_number(sample.get('answer', ''))
            gt = extract_number(sample.get('ground_truth', ''))
            score = sample.get('self_match_score', 0.0)

            if pred is not None and gt is not None:
                total += 1
                is_correct = abs(pred - gt) < 0.1

                if is_correct:
                    correct += 1

                # Group by score range
                if score < 0.3:
                    range_key = '[0.0, 0.3)'
                elif score < 0.5:
                    range_key = '[0.3, 0.5)'
                elif score < 0.7:
                    range_key = '[0.5, 0.7)'
                elif score < 0.9:
                    range_key = '[0.7, 0.9)'
                else:
                    range_key = '[0.9, 1.0]'

                score_ranges[range_key]['total'] += 1
                if is_correct:
                    score_ranges[range_key]['correct'] += 1

                all_scores.append(score)

        # Print accuracy
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"\nOverall H-UAV Accuracy: {correct}/{total} = {accuracy:.1f}%")

        # Print score statistics
        if all_scores:
            print(f"\nSelf-matching Score Statistics:")
            print(f"  Mean:   {np.mean(all_scores):.4f}")
            print(f"  Std:    {np.std(all_scores):.4f}")
            print(f"  Min:    {np.min(all_scores):.4f}")
            print(f"  Max:    {np.max(all_scores):.4f}")
            print(f"  Median: {np.median(all_scores):.4f}")

        # Print accuracy by score range
        print(f"\nAccuracy by Self-matching Score Range:")
        print(f"{'Score Range':<15} {'Samples':<10} {'Correct':<10} {'Accuracy':<10}")
        print("-" * 50)

        for range_key in sorted(score_ranges.keys()):
            stats = score_ranges[range_key]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"{range_key:<15} {stats['total']:<10} {stats['correct']:<10} {acc:.1f}%")

    # Analyze local samples for comparison
    if local_samples:
        print("\n" + "=" * 70)
        print("Local Inference Analysis (for comparison)")
        print("=" * 70)

        correct = 0
        total = 0

        for sample in local_samples:
            pred = extract_number(sample.get('answer', ''))
            gt = extract_number(sample.get('ground_truth', ''))

            if pred is not None and gt is not None:
                total += 1
                if abs(pred - gt) < 0.1:
                    correct += 1

        accuracy = correct / total * 100 if total > 0 else 0
        print(f"\nLocal Accuracy: {correct}/{total} = {accuracy:.1f}%")

    # Print some example errors
    print("\n" + "=" * 70)
    print("Example H-UAV Errors (first 5)")
    print("=" * 70)

    error_count = 0
    for sample in huav_samples[:50]:  # Check first 50
        if error_count >= 5:
            break

        pred = extract_number(sample.get('answer', ''))
        gt = extract_number(sample.get('ground_truth', ''))

        if pred is not None and gt is not None and abs(pred - gt) > 0.1:
            score = sample.get('self_match_score', 0.0)
            print(f"\nQuestion: {sample.get('question', 'N/A')[:80]}...")
            print(f"  Predicted: {pred}")
            print(f"  Ground Truth: {gt}")
            print(f"  Self-match Score: {score:.4f}")
            error_count += 1

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)

if __name__ == "__main__":
    import sys

    # Default path
    results_file = "./outputs/hierarchical_uav/task1_l-uav_results.jsonl"

    if len(sys.argv) > 1:
        results_file = sys.argv[1]

    print(f"Analyzing results from: {results_file}\n")

    try:
        analyze_huav_results(results_file)
    except FileNotFoundError:
        print(f"Error: Results file not found: {results_file}")
        print("\nUsage: python analyze_huav_memory_quality.py [results_file]")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
