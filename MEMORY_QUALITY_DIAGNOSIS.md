# H-UAV Memory Quality Diagnosis and Fixes

## Problem Summary

L-UAV successfully retrieves and applies H-UAV memory, but **accuracy is 0% for cache-hit samples**. This document diagnoses the root causes and provides fixes.

## Key Observations

### 1. Extremely Low Self-Matching Scores
- All `self_match_score` values are between 0.48-0.53
- This is essentially **random matching** (0.5 = random)
- Indicates L-UAV is retrieving **irrelevant memories**

### 2. H-UAV May Be Giving Wrong Answers
- H-UAV tends to predict common values (4 doors, 5 seats)
- May not be accurately reading high-altitude aerial images
- Need to check H-UAV's standalone accuracy

## Diagnostic Steps

### Step 1: Analyze Memory Quality

Run the analysis script:

```bash
cd /home/hk/Downloads/JiaoChen/AirSpatialBot

python analyze_huav_memory_quality.py ./outputs/hierarchical_uav/task1_l-uav_results.jsonl
```

This will show:
- H-UAV accuracy by self-matching score range
- Distribution of self-matching scores
- Example errors with their scores

### Step 2: Check H-UAV's Standalone Accuracy

Check H-UAV's direct results:

```bash
# Look at H-UAV's own predictions
head -20 ./outputs/hierarchical_uav/task1_h-uav_results.jsonl
```

Calculate H-UAV's accuracy to see if the problem is:
- **H-UAV predicting incorrectly** (need to improve H-UAV model)
- **Memory retrieval matching wrong samples** (need to improve self-matching)

### Step 3: Analyze Self-Matching Distribution

Low scores (0.48-0.53) suggest one of:

1. **Query embeddings are too similar** - All questions look the same to the model
2. **Memory embeddings are not discriminative** - Can't distinguish between different samples
3. **Similarity metric is broken** - Cosine similarity not working correctly

## Potential Fixes

### Fix 1: Add Self-Matching Threshold

Only use H-UAV memory when confidence is high.

**File**: `hierarchical_uav/communication/grpc_client.py`

Add threshold check:

```python
# After receiving memory from H-UAV
if self_match_score < 0.7:  # Threshold for high confidence
    logger.warning(f"Low self-match score ({self_match_score:.4f}), skipping H-UAV memory")
    return None  # Don't use memory, fall back to local inference
```

### Fix 2: Improve Query Embedding

The current query embedding might be too generic. Try:

**File**: `hierarchical_uav/eval_task1.py`

Use more discriminative features:
```python
# Include both text and visual features in query
query_text_embedding = ...  # Current text embedding
query_visual_embedding = ...  # Add visual features from the cropped region

# Combine them
query_embedding = torch.cat([query_text_embedding, query_visual_embedding], dim=-1)
```

### Fix 3: Adjust Memory Weight

Current `memory_weight=0.5` might be too high for low-confidence matches.

**File**: `hierarchical_uav/eval_task1.py`

Make weight proportional to confidence:

```python
# Dynamic memory weight based on self-match score
if self_match_score > 0.7:
    memory_weight = 0.8  # High confidence - trust memory
elif self_match_score > 0.5:
    memory_weight = 0.3  # Medium confidence - use lightly
else:
    memory_weight = 0.0  # Low confidence - ignore memory
```

### Fix 4: Check H-UAV Model Quality

If H-UAV's standalone accuracy is also low, the issue is with H-UAV itself:

1. **Fine-tune H-UAV** on the aerial vehicle dataset
2. **Use better vision encoder** for high-altitude images
3. **Adjust H-UAV prompts** to be more specific

## Expected Behavior

After fixes, you should see:

### Good Self-Matching Scores
```
Score Range      Samples    Accuracy
[0.0, 0.3)       0          N/A
[0.3, 0.5)       0          N/A
[0.5, 0.7)       5          40%      ← Low confidence, low accuracy
[0.7, 0.9)       15         75%      ← High confidence, good accuracy
[0.9, 1.0]       13         85%      ← Very high confidence, best accuracy
```

### Improved Cache Hit Accuracy
- High-confidence cache hits: **70-90% accuracy**
- Low-confidence samples fallback to local inference
- Overall system performance improves

## Quick Test

Add this to `eval_task1.py` to print diagnostics:

```python
# After getting memory_value and self_match_score
if memory_value is not None:
    logger.info(f"Sample {i}: self_match_score={self_match_score:.4f}")

    # Check if this will likely be correct
    if self_match_score < 0.6:
        logger.warning(f"  ⚠️  Low confidence - may be incorrect match")
    elif self_match_score < 0.8:
        logger.info(f"  ⚡ Medium confidence")
    else:
        logger.info(f"  ✓ High confidence - likely good match")
```

## Next Steps

1. Run `analyze_huav_memory_quality.py` to get detailed statistics
2. Check if H-UAV standalone accuracy is acceptable (>70%)
3. Implement Fix 1 (threshold) as an immediate improvement
4. Monitor accuracy by score range to validate fixes

## Questions to Answer

1. **What is H-UAV's standalone accuracy?** (Check H-UAV results directly)
2. **Are all self-match scores uniformly low?** (Check distribution)
3. **Do higher scores correlate with better accuracy?** (Check by range)
4. **What are the most common errors?** (Check error examples)

These will guide which fixes to prioritize.
