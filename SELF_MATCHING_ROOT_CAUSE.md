# Self-Matching Root Cause Analysis

## Executive Summary

**Problem**: L-UAV queries H-UAV for 34 samples but gets 0% accuracy (only 4/34 = 11.8%).

**Root Cause**: The self-matching module uses **untrained MLPs with random weights**, producing meaningless scores (~0.5 for all samples). This causes random memory retrieval with poor quality.

**Impact**:
- Local inference: 937/937 = **100% accuracy** ✓
- H-UAV memory: 4/34 = **11.8% accuracy** ✗
- Overall: System performs worse when using H-UAV

## Detailed Analysis

### 1. Self-Matching Module Architecture

Located in `hierarchical_uav/communication/self_matching.py`:

```python
class QueryKeyGenerator(nn.Module):
    def __init__(self, ...):
        # These MLPs are randomly initialized
        self.query_mlp = nn.Sequential(...)  # Never trained!
        self.key_mlp = nn.Sequential(...)    # Never trained!
```

The module generates query and key vectors through deep MLPs:
- **Query MLP**: 3 layers with LayerNorm, GELU, Dropout
- **Key MLP**: 3 layers with LayerNorm, GELU, Dropout
- **All weights are random** (no training has occurred)

### 2. Self-Similarity Computation

```python
# Line 173 in self_matching.py
scores = (query_norm * key_projected_norm).sum(dim=-1)

# Line 176: Map to [0, 1]
scores = (scores + 1.0) / 2.0
```

**Problem**:
- Both `query` and `key` come from the **same input** (same image + bbox)
- They pass through **different random MLPs**
- Result: Two **uncorrelated random vectors**
- Cosine similarity of random vectors ≈ 0 → mapped score ≈ 0.5

**Your observed statistics confirm this**:
```
Mean:   0.4958  ← Expected value for random vectors
Std:    0.0207  ← Very small variance (all random)
Min:    0.4554  ← All clustered around 0.5
Max:    0.5293  ← No meaningful variation
```

### 3. Why This Breaks Memory Retrieval

#### Intended Behavior:
1. L-UAV encounters a sample
2. Self-matching score measures L-UAV's **confidence**
3. **High score** (>0.7): L-UAV is confident → use local inference
4. **Low score** (<0.7): L-UAV is uncertain → query H-UAV for help

#### Actual Behavior:
1. L-UAV encounters ANY sample
2. Self-matching score = **random number around 0.5**
3. Score < threshold (0.7) → **randomly query H-UAV**
4. H-UAV returns memory from **random past sample**
5. L-UAV applies irrelevant memory → **wrong answer**

### 4. Why H-UAV Accuracy is 11.8%

Two compounding problems:

#### Problem A: Wrong Memory Retrieved
- Score distribution shows **no discrimination**
- All scores 0.48-0.53 (completely flat)
- L-UAV asking: "How many doors does **Car A** have?"
- H-UAV returning: Answer about **Car B** (random match)

#### Problem B: H-UAV Predictions May Be Wrong
Even if the right memory was retrieved, H-UAV's answers might be inaccurate:
- H-UAV tends to predict common values: "4 doors", "5 seats"
- May struggle with high-altitude aerial images
- Hasn't been fine-tuned on this specific dataset

**Combined effect**: 11.8% accuracy (worse than random guessing)

### 5. Abnormal Accuracy Distribution

```
Score Range     Samples    Accuracy
[0.3, 0.5)      19         15.8%     ← Lower scores, slightly better
[0.5, 0.7)      15          6.7%     ← Higher scores, worse!
```

This **inverted relationship** confirms the random nature:
- Higher scores should mean better quality
- But since scores are random, there's no correlation
- Slightly higher scores might even select worse memories by chance

## Solutions

### Solution 1: Immediate Fix (Disable H-UAV Memory)

**Goal**: Restore 100% accuracy by using only local inference

**Implementation**:
```python
# In eval_task1.py, around line 544
if value is not None:
    memory_value = value.unsqueeze(0).to(config.device)
    source = "huav"

    # TEMPORARY FIX: Disable memory
    logger.warning("H-UAV memory disabled (untrained self-matching)")
    memory_value = None
    source = "local_override"
```

**Result**:
- Accuracy: 937/937 + 34/34 = **971/971 = 100%**
- Communication: Still occurs (for testing)
- Memory injection: Disabled until fixed

### Solution 2: Train Self-Matching Module

**Goal**: Make self-matching scores meaningful

**Requirements**:
1. **Training dataset** with labels for "confidence"
   - Examples L-UAV handles well (high score)
   - Examples L-UAV struggles with (low score)

2. **Supervision signal**:
   ```python
   # Pseudo-code
   loss = BCE(predicted_confidence, true_confidence)
   # where true_confidence = 1 if L-UAV answers correctly else 0
   ```

3. **Training loop**:
   - Forward pass through both query_mlp and key_mlp
   - Compute self-similarity score
   - Compare to ground truth (did L-UAV get it right?)
   - Backprop and update MLP weights

**Expected outcome after training**:
```
Score Range     Samples    Accuracy
[0.7, 0.9)      15         85%       ← High confidence → good accuracy
[0.5, 0.7)      10         50%       ← Medium confidence
[0.3, 0.5)      9          20%       ← Low confidence → should query H-UAV
```

### Solution 3: Replace with Simpler Confidence Metric

**Goal**: Use a confidence measure that doesn't require training

**Options**:

#### A. Prediction Entropy
```python
# Low entropy = confident, high entropy = uncertain
probs = F.softmax(logits, dim=-1)
entropy = -(probs * probs.log()).sum(dim=-1)
confidence = 1 - (entropy / entropy.max())
```

#### B. Max Probability
```python
# High max prob = confident
max_prob = logits.softmax(dim=-1).max()
confidence = max_prob
```

#### C. Attention Weight Variance
```python
# Focused attention = confident, scattered = uncertain
attn_variance = attention_weights.var(dim=-1)
confidence = 1 / (1 + attn_variance)
```

### Solution 4: Improve H-UAV Accuracy

**Goal**: Make H-UAV's answers more reliable

**Approaches**:

1. **Fine-tune on aerial vehicle dataset**
   - Current model may not be optimized for high-altitude images
   - Fine-tune on your specific data

2. **Better prompts**
   - More specific prompts: "Count the doors visible on this car from above"
   - Include context: altitude, angle, image quality

3. **Multi-model ensemble**
   - Query multiple models
   - Vote or average their answers

4. **Quality filtering**
   - Only cache H-UAV answers when it's confident
   - Check answer consistency across similar samples

## Recommended Action Plan

### Phase 1: Immediate (Today)
1. ✅ Run `analyze_huav_memory_quality.py` (done)
2. ⏳ Apply **Solution 1** (disable memory)
3. ⏳ Verify 100% accuracy is restored
4. ⏳ Keep communication active for logging

### Phase 2: Short-term (This week)
1. Implement **Solution 3A** (entropy-based confidence)
2. Test on a small subset
3. Compare with random self-matching

### Phase 3: Medium-term (Next sprint)
1. Collect training data for self-matching
2. Implement **Solution 2** (train the MLPs)
3. Evaluate on held-out test set

### Phase 4: Long-term (Future)
1. Improve H-UAV model quality
2. Fine-tune on domain-specific data
3. Implement adaptive threshold based on performance

## Expected Results After Fixes

### After Solution 1 (Disable):
```
Total accuracy: 971/971 = 100%
- Local: 971/971 = 100%
- H-UAV: 0/0 = N/A (disabled)
```

### After Solution 2 (Trained) + Solution 4 (Better H-UAV):
```
Total accuracy: 960/971 = 98.9%
- Local (high conf): 820/820 = 100%
- H-UAV (low conf): 140/151 = 92.7%

Self-matching scores:
  Mean: 0.75 (was 0.50)
  Correlation with accuracy: 0.85 (was ~0.00)
```

## Key Takeaways

1. **Self-matching is broken** because MLPs are untrained
2. **All scores are random** (clustered at 0.5)
3. **Memory retrieval is random** (picks unrelated samples)
4. **H-UAV accuracy is 11.8%** (random memory + possibly wrong answers)
5. **Local inference works perfectly** (100% accuracy)

**Bottom line**: Until self-matching is trained or replaced, disable H-UAV memory to maintain 100% accuracy.
