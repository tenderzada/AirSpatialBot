#!/usr/bin/env python3
"""
Test script to verify self-matching mechanism fixes

Run this to verify that:
1. Self-matching scores are no longer constant (1.0)
2. Scores show reasonable distribution
3. Query/key projection works correctly
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hierarchical_uav.communication.self_matching import SelfMatchingModule


def test_self_matching():
    print("=" * 60)
    print("Testing Self-Matching Mechanism")
    print("=" * 60)

    # Create module
    sm_module = SelfMatchingModule(
        feature_dim=512,
        query_dim=128,
        key_dim=256,
        threshold=0.7
    )

    print("\n1. Testing with diverse random features...")
    batch_size = 100
    scores_list = []

    for i in range(10):
        # Create diverse features
        image_features = torch.randn(batch_size, 512)
        bbox_3d = torch.randn(batch_size, 7)

        query, scores, should_query = sm_module(image_features, bbox_3d)
        scores_list.extend(scores.tolist())

    print(f"   Generated {len(scores_list)} scores")

    # Get statistics
    stats = sm_module.get_statistics()

    print("\n2. Score Statistics:")
    print(f"   Mean:  {stats['score_mean']:.4f}")
    print(f"   Std:   {stats['score_std']:.4f}")
    print(f"   Min:   {stats['score_min']:.4f}")
    print(f"   Max:   {stats['score_max']:.4f}")

    print("\n3. Query Statistics:")
    print(f"   Total queries:  {stats['total_queries']}")
    print(f"   H-UAV queries:  {stats['huav_queries']}")
    print(f"   Query rate:     {stats['query_rate']:.2%}")

    # Check if scores are diverse
    print("\n4. Score Distribution:")
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    for i in range(len(bins)-1):
        count = sum(1 for s in scores_list if bins[i] <= s < bins[i+1])
        pct = count / len(scores_list) * 100
        print(f"   [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:4d} ({pct:5.1f}%)")

    # Validation
    print("\n" + "=" * 60)
    print("Validation Results:")
    print("=" * 60)

    passed = True

    # Check 1: Scores are not all 1.0
    if stats['score_std'] < 0.001:
        print("❌ FAILED: Scores are constant (std too low)")
        passed = False
    else:
        print(f"✓ PASSED: Scores show variation (std={stats['score_std']:.4f})")

    # Check 2: Scores are in valid range [0, 1]
    if stats['score_min'] < 0 or stats['score_max'] > 1:
        print(f"❌ FAILED: Scores out of range [{stats['score_min']:.4f}, {stats['score_max']:.4f}]")
        passed = False
    else:
        print(f"✓ PASSED: Scores in valid range [0, 1]")

    # Check 3: Some queries should trigger H-UAV (with threshold 0.7)
    if stats['huav_queries'] == 0:
        print(f"⚠️  WARNING: No H-UAV queries triggered (may need threshold adjustment)")
        print(f"   Current threshold: 0.7, Max score: {stats['score_max']:.4f}")
    else:
        print(f"✓ PASSED: H-UAV queries triggered ({stats['query_rate']:.1%} query rate)")

    # Check 4: Query/key projection works
    print("\n5. Testing query/key projection...")
    test_features = torch.randn(1, 512)
    test_bbox = torch.randn(1, 7)
    query, key = sm_module.qk_generator(test_features, test_bbox)

    if query.shape == (1, 128) and key.shape == (1, 256):
        print(f"✓ PASSED: Query/key dimensions correct (query={query.shape}, key={key.shape})")
    else:
        print(f"❌ FAILED: Query/key dimensions incorrect (query={query.shape}, key={key.shape})")
        passed = False

    # Check 5: Test auto-adjust threshold
    print("\n6. Testing auto-adjust threshold...")
    original_threshold = sm_module.gate.adaptive_threshold.item()
    new_threshold = sm_module.auto_adjust_threshold(target_query_rate=0.3)
    print(f"   Original threshold: {original_threshold:.4f}")
    print(f"   Adjusted threshold: {new_threshold:.4f}")
    print(f"   Adjustment: {new_threshold - original_threshold:+.4f}")

    if new_threshold != original_threshold:
        print(f"✓ PASSED: Threshold auto-adjustment works")
    else:
        print(f"⚠️  WARNING: Threshold unchanged (may be expected if score distribution is ideal)")

    print("\n" + "=" * 60)
    if passed:
        print("✅ All tests PASSED!")
    else:
        print("❌ Some tests FAILED - please review the output above")
    print("=" * 60)

    return passed


if __name__ == "__main__":
    success = test_self_matching()
    sys.exit(0 if success else 1)
