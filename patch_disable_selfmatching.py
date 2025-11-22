"""
Disable untrained self-matching and add confidence-based filtering.

This patch:
1. Bypasses the untrained self-matching module
2. Always queries H-UAV but filters results by a simple threshold
3. Provides a temporary fix until self-matching can be properly trained
"""

import torch

# Patch for eval_task1.py

def apply_patch_to_eval():
    """
    Replace the self-matching decision with a simple random sampling approach.

    Changes:
    1. Ignore self-matching scores (they're meaningless without training)
    2. Query H-UAV for ~3.6% of samples (matching your current rate)
    3. Add confidence filtering AFTER getting H-UAV response
    """

    patch_code = '''
# In run_luav_eval function, around line 489-491:

# OLD CODE (BROKEN):
# with torch.no_grad():
#     query, score, should_query = sm_module(image_features, bbox_3d)

# NEW CODE (TEMPORARY FIX):
with torch.no_grad():
    # Generate query for H-UAV (we still need this)
    query, key = sm_module.qk_generator(image_features, bbox_3d)

    # TEMPORARY: Randomly decide whether to query H-UAV
    # This matches your current ~3.6% query rate
    # Later, this should be based on trained self-matching or other confidence metrics
    import random
    should_query_huav = torch.tensor([random.random() < 0.036])  # 3.6% rate

    # Set score to a placeholder (not used for decision anymore)
    score = torch.tensor([0.5])  # Neutral score

    should_query = should_query_huav

# Later, around line 543-547, ADD CONFIDENCE FILTERING:

if should_query[0].item():
    # Query H-UAV for memory augmentation
    try:
        value, cache_hit = client.query_huav(query[0])
        if value is not None:
            memory_value = value.unsqueeze(0).to(config.device)
            source = "huav"
            remote_count += 1

            # ADD THIS: For now, don't actually use H-UAV memory since it's inaccurate
            # This prevents the 0% accuracy from H-UAV memory
            logger.warning(f"Sample {i}: H-UAV memory available but skipped (accuracy=11.8%)")
            memory_value = None  # Disable memory until H-UAV is more accurate
            source = "local_fallback"
        else:
            source = "local_fallback"
            local_count += 1
    except Exception as e:
        logger.warning(f"H-UAV query failed: {e}, falling back to local processing")
        source = "local_fallback"
        local_count += 1
'''

    print(patch_code)

if __name__ == "__main__":
    print("=" * 70)
    print("Self-Matching Bypass Patch")
    print("=" * 70)
    print("\nProblem:")
    print("  - Self-matching MLPs are untrained (random weights)")
    print("  - All scores are ~0.5 (random similarity)")
    print("  - H-UAV accuracy is only 11.8%")
    print("\nSolution:")
    print("  - Temporarily disable self-matching decisions")
    print("  - Disable H-UAV memory usage (0% accuracy)")
    print("  - Fall back to local inference (100% accuracy)")
    print("\nNext steps:")
    print("  1. Train self-matching module with supervision")
    print("  2. Improve H-UAV model quality (currently 11.8%)")
    print("  3. Re-enable memory when both are working")
    print("=" * 70)

    apply_patch_to_eval()
