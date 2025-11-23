#!/usr/bin/env python3
"""
Debug script to identify which samples have different KB coverage
between L-UAV and H-UAV standalone evaluations.
"""

import json
import sys

def load_results(filepath):
    """Load JSONL results."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def main():
    if len(sys.argv) != 3:
        print("Usage: python debug_kb_difference.py <luav_results.jsonl> <huav_results.jsonl>")
        sys.exit(1)

    luav_file = sys.argv[1]
    huav_file = sys.argv[2]

    print("Loading results...")
    luav_results = load_results(luav_file)
    huav_results = load_results(huav_file)

    print(f"L-UAV samples: {len(luav_results)}")
    print(f"H-UAV samples: {len(huav_results)}")
    print()

    # Create mappings by question_id
    luav_by_qid = {r['question_id']: r for r in luav_results}
    huav_by_qid = {r['question_id']: r for r in huav_results}

    # Find samples with different sources
    different_source = []
    luav_kb_only = []
    huav_vlm_only = []

    for qid in luav_by_qid:
        if qid not in huav_by_qid:
            print(f"Warning: Question {qid} only in L-UAV results")
            continue

        luav_r = luav_by_qid[qid]
        huav_r = huav_by_qid[qid]

        luav_source = luav_r.get('source', 'unknown')
        huav_source = huav_r.get('source', 'unknown')

        if luav_source != huav_source:
            different_source.append({
                'question_id': qid,
                'image_id': luav_r['image_id'],
                'qtype': luav_r.get('qtype', 'unknown'),
                'luav_source': luav_source,
                'huav_source': huav_source,
                'luav_answer': luav_r.get('answer', ''),
                'huav_answer': huav_r.get('answer', ''),
                'ground_truth': luav_r.get('ground_truth', ''),
                'question': luav_r.get('question', '')[:60]
            })

            if luav_source == 'knowledge_base' and huav_source == 'huav_vlm':
                luav_kb_only.append(different_source[-1])
            elif luav_source == 'local' and huav_source == 'huav_vlm':
                # This is expected - both use VLM for unknown samples
                pass

    print("=" * 80)
    print(f"SAMPLES WITH DIFFERENT SOURCES: {len(different_source)}")
    print("=" * 80)
    print()

    # Analyze KB → VLM transitions (the problematic ones)
    if luav_kb_only:
        print(f"⚠ CRITICAL: {len(luav_kb_only)} samples changed from KB (L-UAV) to VLM (H-UAV)")
        print("These should have been answered by KB in H-UAV but weren't!")
        print()

        print("Breakdown by question type:")
        qtype_counts = {}
        for sample in luav_kb_only:
            qtype = sample['qtype']
            qtype_counts[qtype] = qtype_counts.get(qtype, 0) + 1

        for qtype, count in sorted(qtype_counts.items()):
            print(f"  {qtype:12s}: {count}")
        print()

        # Show first 10 examples
        print("First 10 examples:")
        print("-" * 80)
        for i, sample in enumerate(luav_kb_only[:10]):
            gt = sample['ground_truth']
            luav_ans = sample['luav_answer']
            huav_ans = sample['huav_answer']

            luav_correct = str(gt).lower() in str(luav_ans).lower()
            huav_correct = str(gt).lower() in str(huav_ans).lower()

            print(f"\n[{i+1}] Question ID: {sample['question_id']}")
            print(f"    Image: {sample['image_id']}")
            print(f"    Type: {sample['qtype']}")
            print(f"    Question: {sample['question']}")
            print(f"    Ground Truth: {gt}")
            print(f"    L-UAV (KB): {luav_ans} {'✓' if luav_correct else '✗'}")
            print(f"    H-UAV (VLM): {huav_ans} {'✓' if huav_correct else '✗'}")

    # Check if there are any other patterns
    print()
    print("=" * 80)
    print("SOURCE DISTRIBUTION COMPARISON")
    print("=" * 80)

    luav_sources = {}
    huav_sources = {}

    for r in luav_results:
        src = r.get('source', 'unknown')
        luav_sources[src] = luav_sources.get(src, 0) + 1

    for r in huav_results:
        src = r.get('source', 'unknown')
        huav_sources[src] = huav_sources.get(src, 0) + 1

    print(f"\n{'Source':<20s} {'L-UAV':<15s} {'H-UAV':<15s} {'Difference':<10s}")
    print("-" * 60)

    all_sources = set(luav_sources.keys()) | set(huav_sources.keys())
    for src in sorted(all_sources):
        l_count = luav_sources.get(src, 0)
        h_count = huav_sources.get(src, 0)
        diff = h_count - l_count
        print(f"{src:<20s} {l_count:4d} ({l_count/len(luav_results)*100:5.1f}%)  "
              f"{h_count:4d} ({h_count/len(huav_results)*100:5.1f}%)  {diff:+4d}")

if __name__ == '__main__':
    main()
