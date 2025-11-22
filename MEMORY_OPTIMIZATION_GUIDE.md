# Memory System Optimization Guide

## Executive Summary

**Current Status**:
- Memory injection mechanism: ✅ Working correctly
- Local inference accuracy: ✅ 100% (937/937)
- H-UAV memory accuracy: ❌ 11.8% (4/34)

**Root Causes**:
1. **Untrained self-matching** → Random memory retrieval
2. **Low H-UAV quality** → Even correct retrieval gives wrong answers

**Solution**: Optimize both components instead of disabling memory

---

## Optimization Strategy Overview

We have created 3 major optimization approaches:

### 1. Train Self-Matching Module ⭐ **Priority**
- **File**: `hierarchical_uav/train_self_matching.py`
- **Goal**: Make self-matching scores meaningful
- **Expected improvement**: Scores correlate with accuracy (AUC > 0.8)

### 2. Improve H-UAV Quality
- **File**: `hierarchical_uav/improve_huav_quality.py`
- **Goal**: Boost H-UAV accuracy from 11.8% → 70%+
- **Methods**: Better prompts, validation, consistency checks

### 3. Dynamic Memory Weighting
- **File**: `hierarchical_uav/dynamic_memory_weight.py`
- **Goal**: Adaptive weight based on confidence
- **Benefit**: Graceful degradation for uncertain memories

---

## 📊 Optimization 1: Train Self-Matching Module

### Problem
Current self-matching MLPs are **randomly initialized** and never trained:
```python
# hierarchical_uav/communication/self_matching.py
self.query_mlp = nn.Sequential(...)  # Random weights!
self.key_mlp = nn.Sequential(...)    # Random weights!
```

This causes all scores to cluster at 0.5 (random similarity).

### Solution: Supervised Training

#### Step 1: Extract Features
```bash
cd /home/hk/Downloads/JiaoChen/AirSpatialBot

# Extract image features for training
python hierarchical_uav/train_self_matching.py \
    --mode extract \
    --dataset ./data/task1_dataset.json \
    --features ./outputs/hierarchical_uav/training_features.pt \
    --device cuda:0
```

This processes each sample and saves:
- Image features from L-UAV's vision encoder
- 3D bounding box parameters
- Links to evaluation results (for correctness labels)

#### Step 2: Train Self-Matching
```bash
# Train the MLPs with supervision
python hierarchical_uav/train_self_matching.py \
    --mode train \
    --features ./outputs/hierarchical_uav/training_features.pt \
    --results ./outputs/hierarchical_uav/task1_l-uav_results.jsonl \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --device cuda:0
```

**Training objective**:
```
Loss = BCE(self_match_score, correctness_label)

Where:
- self_match_score = Model's confidence [0, 1]
- correctness_label = 1 if L-UAV answered correctly, 0 otherwise
```

#### Step 3: Load Trained Weights
```python
# In eval_task1.py, after creating sm_module
sm_module = SelfMatchingModule(...)

# Load trained weights
checkpoint = torch.load('./outputs/hierarchical_uav/best_self_matching.pt')
sm_module.load_state_dict(checkpoint)

print("✓ Loaded trained self-matching weights")
```

### Expected Results After Training

**Before** (untrained):
```
Score distribution: Mean=0.50, Std=0.02, Range=[0.46-0.53]
Accuracy correlation: ~0.0 (random)
```

**After** (trained):
```
Score distribution: Mean=0.72, Std=0.15, Range=[0.35-0.95]
Accuracy correlation: ~0.85 (strong positive)

Score Range     Samples    Accuracy
[0.7, 1.0)      420        95%      ← High scores → high accuracy
[0.5, 0.7)      320        72%      ← Medium scores
[0.0, 0.5)      231        45%      ← Low scores → query H-UAV
```

---

## 🎯 Optimization 2: Improve H-UAV Quality

### Current Issues

**H-UAV accuracy = 11.8%** because:
1. Generic prompts don't leverage aerial image context
2. No validation of answers
3. No consistency checking
4. Predicts common values ("4 doors", "5 seats") regardless of actual vehicle

### Solution: Quality Enhancement Pipeline

#### Enhancement 1: Context-Aware Prompts

**Before** (generic):
```python
question = "How many doors does this car have?"
```

**After** (context-aware):
```python
from hierarchical_uav.improve_huav_quality import HUAVQualityImprover

improver = HUAVQualityImprover()

# Build rich context
bbox_info = {
    'AGL': sample['AGL'],           # Altitude
    'pitch_angle': sample['pitch_angle'],
    'length': sample['length'],
    'width': sample['width'],
    'height': sample['height']
}

# Generate improved prompt
improved_prompt = improver.improve_prompt(
    question=question,
    qtype=qtype,
    bbox_info=bbox_info
)

# Example output:
# "This is an aerial image taken from 20m altitude at 60° angle.
#  The highlighted vehicle is a mid-size vehicle (sedan or crossover)
#  with dimensions approximately 4.5m × 1.8m.
#  Based on the vehicle's size and shape, how many doors does it have?
#  Common values are 2 (coupe), 4 (sedan), or 5 (hatchback/SUV).
#  Answer with just the number."
```

**Expected improvement**: +20-30% accuracy

#### Enhancement 2: Answer Validation

```python
# After getting H-UAV's answer
validation = improver.validate_answer(
    answer=huav_answer,
    qtype=qtype,
    confidence_score=self_match_score
)

if not validation['is_valid']:
    # Reject invalid answer, use local inference instead
    logger.warning(f"H-UAV answer '{huav_answer}' failed validation")
    use_memory = False
else:
    # Use validated (possibly corrected) answer
    huav_answer = validation['validated_answer']
    confidence = validation['confidence']
```

**Validation rules**:
- `doors`: Must be in [2, 3, 4, 5]
- `seats`: Must be in [2, 4, 5, 6, 7, 8, 9]
- `brand`: Must be known manufacturer or fuzzy match

**Expected improvement**: +10-15% accuracy

#### Enhancement 3: Consistency Checking

```python
# Check if answers are consistent with each other
consistency = improver.apply_consistency_check(
    answer=huav_answer_doors,
    qtype='doors',
    related_answers={'seats': huav_answer_seats}
)

if not consistency['is_consistent']:
    # Reduce confidence or flag for review
    logger.warning(f"Inconsistent: {consistency['conflicts']}")
    confidence *= 0.5
```

**Rules**:
- 5 doors → Usually 5+ seats (hatchback/SUV)
- 2 doors → Usually 2-4 seats (coupe/convertible)
- Large vehicle (>5m) → Usually 6+ seats

**Expected improvement**: +5-10% accuracy

### Integration Example

```python
# In H-UAV server (hierarchical_uav/run_huav.py)
from hierarchical_uav.improve_huav_quality import apply_quality_improvements

# After H-UAV generates answer
improved_result = apply_quality_improvements(
    huav_answer=raw_answer,
    question=question,
    qtype=qtype,
    self_match_score=self_match_score,
    bbox_info=sample_info,
    related_answers=previous_answers  # From same vehicle
)

if improved_result['answer'] is None:
    # Quality check failed
    logger.info(f"Answer rejected: {improved_result['reason']}")
    return None  # L-UAV will use local inference
else:
    # Use improved answer
    final_answer = improved_result['answer']
    confidence = improved_result['confidence']
```

### Expected Overall H-UAV Accuracy

| Enhancement | Accuracy Gain | Cumulative |
|-------------|---------------|------------|
| Baseline | - | 11.8% |
| Better prompts | +25% | ~37% |
| Answer validation | +15% | ~52% |
| Consistency checks | +8% | ~60% |
| **+ Fine-tuning on dataset** | +20% | **~80%** |

---

## ⚖️ Optimization 3: Dynamic Memory Weighting

### Problem
Current implementation uses **fixed weight = 0.5**:
```python
output_ids = luav_model.generate_with_memory(
    memory_weight=0.5,  # Always 50-50 blend
    ...
)
```

This doesn't account for:
- Memory relevance (self-match score)
- H-UAV confidence
- L-UAV's own confidence

### Solution: Adaptive Weighting

```python
from hierarchical_uav.dynamic_memory_weight import DynamicMemoryWeightAdjuster

# Initialize (once)
weight_adjuster = DynamicMemoryWeightAdjuster(
    base_weight=0.5,
    min_weight=0.1,
    max_weight=0.9
)

# Inside evaluation loop
if memory_value is not None:
    # Compute dynamic weight
    weight_result = weight_adjuster.compute_weight(
        self_match_score=score[0].item(),
        huav_confidence=None,  # Optional: from H-UAV
        luav_confidence=None   # Optional: from L-UAV logits
    )

    memory_weight = weight_result['weight']

    logger.info(
        f"Sample {i}: self_match={score[0].item():.3f}, "
        f"weight={memory_weight:.3f}"
    )

    # Use dynamic weight
    output_ids = luav_model.generate_with_memory(
        memory_weight=memory_weight,  # Adaptive!
        ...
    )
```

### Weight Decision Logic

| Self-Match Score | H-UAV Conf | Recommended Weight | Rationale |
|------------------|------------|-------------------|-----------|
| > 0.8 | > 0.8 | 0.9 | High relevance, confident H-UAV |
| > 0.7 | > 0.7 | 0.7 | Good match, trust memory |
| 0.5-0.7 | 0.5-0.7 | 0.4 | Moderate confidence |
| < 0.5 | Any | 0.1 | Low relevance, ignore memory |
| Any | < 0.5 | 0.2 | H-UAV uncertain |

### Expected Benefits

**Scenario 1**: High-quality memory (score=0.85)
- Fixed weight: 0.5 → Underutilizes good memory
- Dynamic weight: 0.8 → Properly leverages memory
- **Result**: Better accuracy on hard samples

**Scenario 2**: Low-quality memory (score=0.45)
- Fixed weight: 0.5 → Corrupts answer with bad memory
- Dynamic weight: 0.1 → Mostly ignores bad memory
- **Result**: Maintains local accuracy

**Overall improvement**: +15-25% on samples that use memory

---

## 📅 Implementation Roadmap

### Phase 1: Immediate (Today) - **Dynamic Weighting**

**Goal**: Prevent bad memory from hurting accuracy

```bash
# 1. Pull latest code
git pull origin claude/add-uav-evaluation-stats-018vmWqi4RkwFL8znAUZFexN

# 2. Edit eval_task1.py to use dynamic weights
# Add around line 580:
from hierarchical_uav.dynamic_memory_weight import DynamicMemoryWeightAdjuster

weight_adjuster = DynamicMemoryWeightAdjuster()

# Then in the loop:
weight_result = weight_adjuster.compute_weight(score[0].item())
memory_weight = weight_result['weight']

# 3. Re-run evaluation
./run_huav.sh  # Terminal 1
./run_luav.sh  # Terminal 2
```

**Expected result**:
- Cache-hit accuracy: 11.8% → ~30-40%
- Overall accuracy: ~97% (no degradation from bad memory)

---

### Phase 2: Short-term (This Week) - **H-UAV Quality**

**Goal**: Improve H-UAV answers with better prompts and validation

```bash
# 1. Integrate quality improvements into H-UAV server
# Edit hierarchical_uav/run_huav.py

from hierarchical_uav.improve_huav_quality import apply_quality_improvements

# After generating answer, add:
improved = apply_quality_improvements(...)
if improved['answer'] is None:
    return None  # Reject low-quality answer

# 2. Test on small subset
# Run eval on first 100 samples to verify improvements

# 3. Full evaluation
# If results are good, run on full dataset
```

**Expected result**:
- H-UAV accuracy: 11.8% → ~50-60%
- Cache-hit samples contribute positively

---

### Phase 3: Medium-term (Next 2 Weeks) - **Train Self-Matching**

**Goal**: Make memory retrieval accurate

```bash
# 1. Extract features (one-time)
python hierarchical_uav/train_self_matching.py \
    --mode extract \
    --dataset ./data/task1_dataset.json \
    --features ./outputs/hierarchical_uav/training_features.pt

# 2. Train (takes 2-3 hours on GPU)
python hierarchical_uav/train_self_matching.py \
    --mode train \
    --features ./outputs/hierarchical_uav/training_features.pt \
    --results ./outputs/hierarchical_uav/task1_l-uav_results.jsonl \
    --epochs 50

# 3. Load trained model in eval_task1.py
sm_module.load_state_dict(torch.load('best_self_matching.pt'))

# 4. Re-evaluate
```

**Expected result**:
- Self-match scores meaningful (AUC > 0.85)
- Memory retrieval precision: random → 75%+
- Only query H-UAV when truly uncertain

---

### Phase 4: Long-term (Future) - **H-UAV Fine-tuning**

**Goal**: Specialized H-UAV for aerial vehicles

1. Collect dataset of aerial vehicle images with attributes
2. Fine-tune H-UAV model on this dataset
3. Train with aerial-specific augmentations
4. Evaluate on held-out test set

**Expected result**:
- H-UAV accuracy: 60% → 85%+
- Memory becomes highly valuable resource

---

## 📊 Expected Final Performance

### Current Baseline
```
Total: 971 samples
- Local: 937/937 = 100%
- H-UAV: 4/34 = 11.8%
Overall: 941/971 = 96.9%
```

### After Phase 1 (Dynamic Weights)
```
Total: 971 samples
- Local: 937/937 = 100%
- H-UAV: 12/34 = 35%  (bad memory downweighted)
Overall: 949/971 = 97.7%  (+0.8%)
```

### After Phase 2 (H-UAV Quality)
```
Total: 971 samples
- Local: 937/937 = 100%
- H-UAV: 20/34 = 59%  (better prompts, validation)
Overall: 957/971 = 98.6%  (+1.7%)
```

### After Phase 3 (Trained Self-Matching)
```
Total: 971 samples
- Local: 937/937 = 100%
- H-UAV: 28/34 = 82%  (retrieves relevant memory)
Overall: 965/971 = 99.4%  (+2.5%)
```

### After Phase 4 (Fine-tuned H-UAV)
```
Total: 971 samples
- Local: 937/937 = 100%
- H-UAV: 32/34 = 94%  (domain-specialized model)
Overall: 969/971 = 99.8%  (+2.9%)
```

---

## 🔬 Monitoring and Validation

### Key Metrics to Track

1. **Self-Matching Quality**
   - Score distribution (mean, std, range)
   - Correlation with accuracy (AUC)
   - Query rate (target: 3-5%)

2. **H-UAV Quality**
   - Standalone accuracy
   - Answer rejection rate
   - Consistency violation rate

3. **Memory Impact**
   - Accuracy with memory vs without
   - Average memory weight applied
   - Confidence-stratified performance

### Validation Protocol

After each phase:
```bash
# 1. Run full evaluation
python hierarchical_uav/eval_task1.py ...

# 2. Analyze results
python analyze_huav_memory_quality.py ./outputs/.../results.jsonl

# 3. Check key metrics
# - Did cache-hit accuracy improve?
# - Is self-match score distribution better?
# - Any negative impact on local samples?

# 4. Compare with baseline
# Only deploy if metrics improve
```

---

## 🎓 Key Takeaways

1. **Memory injection works!** The technical implementation is sound.

2. **Two independent problems**:
   - Self-matching needs training (retrieval quality)
   - H-UAV needs improvement (answer quality)

3. **Solve incrementally**:
   - Phase 1 (Dynamic weights): Quick win, no training
   - Phase 2 (H-UAV quality): Medium effort, good gains
   - Phase 3 (Train self-match): Highest ROI, requires data
   - Phase 4 (Fine-tune H-UAV): Long-term, best quality

4. **Don't disable memory!** Optimize it for maximum research value.

---

## 📚 Files Reference

| File | Purpose | Usage |
|------|---------|-------|
| `train_self_matching.py` | Train MLPs | Phase 3 |
| `improve_huav_quality.py` | Better prompts/validation | Phase 2 |
| `dynamic_memory_weight.py` | Adaptive weighting | Phase 1 |
| `analyze_huav_memory_quality.py` | Metrics analysis | All phases |
| `SELF_MATCHING_ROOT_CAUSE.md` | Root cause analysis | Reference |

---

## 🚀 Quick Start

**Start with Phase 1 (easiest):**

```bash
# 1. Get latest code
cd /home/hk/Downloads/JiaoChen/AirSpatialBot
git pull origin claude/add-uav-evaluation-stats-018vmWqi4RkwFL8znAUZFexN

# 2. Test dynamic weights on a few samples
# Edit eval_task1.py to add dynamic weight adjustment

# 3. Re-run evaluation
./run_huav.sh &
./run_luav.sh

# 4. Check results
python analyze_huav_memory_quality.py ./outputs/.../results.jsonl

# Expected: Cache-hit accuracy improves from 11.8% to ~35%
```

**Questions?** Refer to the detailed sections above or the source code comments.

Memory optimization is valuable research - let's make it work! 🎯
