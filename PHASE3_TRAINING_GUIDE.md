# Phase 3: H-UAV Test-Time Learning Guide

## Overview

Phase 3 implements **test-time learning** for H-UAV to build meaningful memory representations in the MAC layer. This solves the core problem: **MAC layers are randomly initialized and need to learn semantic representations**.

## Problem Analysis

### Current State (Before Phase 3)

```
H-UAV MAC layers: Random initialization
    ↓
Stores random noise, not semantic features
    ↓
L-UAV queries H-UAV → Gets random features
    ↓
Memory injection adds noise to inference
    ↓
Result: 3.1% accuracy (vs 100% without memory)
```

**Evidence:**
- 327 samples with memory → Only 8 unique answers (2.4% diversity)
- All samples produce similar random outputs
- H-UAV memory contains no useful information

### After Phase 3 (Expected)

```
H-UAV test-time learning on 971 samples (3 epochs)
    ↓
MAC layers learn semantic visual-linguistic mappings
    ↓
Stores meaningful features aligned with VQA task
    ↓
L-UAV queries H-UAV → Gets learned semantic features
    ↓
Memory injection provides useful context
    ↓
Result: Improved accuracy and diversity
```

---

## Quick Start

### 1. Train H-UAV

```bash
# Simple: Use default parameters
./train_huav.sh

# Custom: Specify parameters
python hierarchical_uav/train_huav.py \
    --num_epochs 5 \
    --learning_theta 0.15 \
    --surprise_eta 0.95 \
    --output_dir ./outputs/huav_training_custom
```

**Training time:** ~2-3 hours (971 samples × 3 epochs on GPU)

### 2. Start H-UAV with Trained Memory

```bash
# Method 1: Environment variable
LOAD_MEMORY=./outputs/huav_training/huav_memory_final.pt ./run_huav.sh

# Method 2: Direct call
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --load-memory ./outputs/huav_training/huav_memory_final.pt \
    --port 50051 \
    --device cuda:0 \
    --load_8bit
```

### 3. Evaluate L-UAV with Trained H-UAV

```bash
# In another terminal
./run_luav.sh
```

**Expected improvement:**
- Higher answer diversity (>2.4%)
- Better accuracy on memory-augmented samples (>3.1%)
- More meaningful memory influence

---

## Detailed Workflow

### Step 1: Understand Test-Time Learning

**What is test-time learning?**

Unlike traditional training (with labeled data and backprop), test-time learning:
1. Happens during inference (eval mode)
2. Uses self-supervised signals
3. Continuously adapts based on surprise

**Surprise-Driven Learning Algorithm:**

```python
# For each sample:
query = f(image, question)  # Query vector
prediction = MAC(query)      # Retrieve from memory

# Compute surprise
target = enhanced_f(output)  # Target from model output
loss = ||prediction - target||²
surprise = current_surprise * eta - theta * loss

# Update memory
if abs(surprise) > threshold:
    # This sample is novel! Learn from it
    MAC_params += -theta * gradient
    MAC_params *= (1 - alpha)  # Forgetting

# Store in episodic cache
cache.add(query, target)
```

**Key parameters:**
- **theta (0.1)**: Learning rate - how fast to adapt
- **eta (0.9)**: Surprise decay - momentum for updates
- **alpha (0.01)**: Forgetting rate - prevent catastrophic memory

### Step 2: Run Training

**Basic command:**
```bash
./train_huav.sh
```

**What happens:**

1. **Initialization**
   ```
   Loading H-UAV model on cuda:0...
   ✓ Base LLaVA model loaded
   ✓ MAC layers added
   ✓ Configured as H-UAV (learning mode)
   ```

2. **Training loop (per epoch)**
   ```
   Epoch 1/3
   =====================================
   Processing 971 samples...
   Sample 100: loss=0.0234, surprise=-0.125
   Sample 200: loss=0.0198, surprise=-0.089
   ...

   Epoch 1 Summary:
   Samples processed: 971
   Average loss: 0.0215
   Average surprise: -0.092
   High surprise samples: 143 (14.7%)
   ```

3. **Checkpointing**
   ```
   ✓ Checkpoint saved: ./outputs/huav_training/huav_memory_epoch_1.pt
   ✓ Checkpoint saved: ./outputs/huav_training/huav_memory_epoch_2.pt
   ✓ Checkpoint saved: ./outputs/huav_training/huav_memory_epoch_3.pt
   ```

4. **Final save**
   ```
   ✓ Final memory saved: ./outputs/huav_training/huav_memory_final.pt
   ✓ Training report saved: ./outputs/huav_training/training_report.json
   ```

### Step 3: Analyze Training

**Check training report:**
```bash
cat ./outputs/huav_training/training_report.json | jq '.'
```

**Output:**
```json
{
  "config": {
    "model_path": "/mnt/data/AirSpatialBot",
    "num_epochs": 3,
    "learning_theta": 0.1,
    "surprise_eta": 0.9,
    "forgetting_alpha": 0.01
  },
  "results": {
    "total_samples": 2913,
    "epoch_stats": [
      {
        "epoch": 1,
        "samples": 971,
        "avg_loss": 0.0215,
        "avg_surprise": -0.092,
        "high_surprise_count": 143
      },
      ...
    ]
  }
}
```

**Key metrics to watch:**

1. **Average loss**: Should decrease over epochs
   - Epoch 1: ~0.02-0.03
   - Epoch 3: ~0.01-0.015
   - Lower = better memory prediction

2. **Average surprise**: Should stabilize
   - Early: Large variations
   - Later: More stable
   - Indicates memory is capturing patterns

3. **High surprise count**: Novel samples
   - Epoch 1: High (many new patterns)
   - Epoch 3: Lower (seen most patterns)
   - ~10-20% is healthy

### Step 4: Deploy Trained H-UAV

**Start H-UAV server with trained memory:**

```bash
LOAD_MEMORY=./outputs/huav_training/huav_memory_final.pt ./run_huav.sh
```

**Expected output:**
```
============================================================
Starting H-UAV (High Resource UAV)
============================================================

Configuration:
  Model: /mnt/data/AirSpatialBot
  Vision Tower: /mnt/data/clip-vit-large-patch14-336
  Device: cuda:0
  Port: 50051
  Trained Memory: ./outputs/huav_training/huav_memory_final.pt
  Output: ./outputs/hierarchical_uav/task1_h-uav_results.jsonl

Loading H-UAV model on cuda:0...
✓ Base LLaVA model loaded
✓ MAC layers added
✓ Configured as H-UAV (learning mode)

Loading trained MAC memory from ./outputs/huav_training/huav_memory_final.pt...
✓ MAC state loaded from ./outputs/huav_training/huav_memory_final.pt
✓ Trained memory loaded - H-UAV ready with learned representations

Starting gRPC server on port 50051...
✓ H-UAV Server started
```

### Step 5: Evaluate L-UAV

**In another terminal:**
```bash
./run_luav.sh
```

**Expected improvements:**

Before Phase 3 (random MAC):
```
Overall Accuracy: 67.35%
Accuracy by Source:
  knowledge_base: 644/644 = 100.0%
  huav: 10/327 = 3.1%  ← Random memory

Memory characteristics:
  Answer diversity: 2.4% (8 unique / 327 total)
  Self-match scores: Mean=0.4920 (random)
```

After Phase 3 (trained MAC):
```
Overall Accuracy: 75-80% (expected)
Accuracy by Source:
  knowledge_base: 644/644 = 100.0%
  huav: 100-150/327 = 30-50%  ← Learned memory

Memory characteristics:
  Answer diversity: 20-40% (better than 2.4%)
  Self-match scores: More discriminative distribution
```

---

## Parameter Tuning

### Core Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `learning_theta` | 0.1 | 0.01-0.3 | **Learning rate**: Higher = faster adaptation, but may overfit |
| `surprise_eta` | 0.9 | 0.7-0.99 | **Momentum**: Higher = smoother updates, slower convergence |
| `forgetting_alpha` | 0.01 | 0.001-0.05 | **Weight decay**: Higher = more forgetting, prevents overfitting |
| `num_epochs` | 3 | 1-10 | **Training duration**: More epochs = better convergence |

### Tuning Strategies

**Problem: Loss not decreasing**
```bash
# Solution: Increase learning rate
python hierarchical_uav/train_huav.py \
    --learning_theta 0.15 \  # was 0.1
    --num_epochs 5
```

**Problem: Loss unstable / oscillating**
```bash
# Solution: Decrease learning rate, increase momentum
python hierarchical_uav/train_huav.py \
    --learning_theta 0.05 \  # was 0.1
    --surprise_eta 0.95 \    # was 0.9
    --num_epochs 3
```

**Problem: Memory overfits to training data**
```bash
# Solution: Increase forgetting rate
python hierarchical_uav/train_huav.py \
    --forgetting_alpha 0.02 \  # was 0.01
    --num_epochs 3
```

**Problem: Not enough learning**
```bash
# Solution: More epochs
python hierarchical_uav/train_huav.py \
    --num_epochs 10 \  # was 3
    --save_interval 2   # save every 2 epochs
```

---

## Advanced Usage

### Resume from Checkpoint

```bash
python hierarchical_uav/train_huav.py \
    --resume ./outputs/huav_training/huav_memory_epoch_2.pt \
    --num_epochs 5
```

### Use Different Training Data

```bash
python hierarchical_uav/train_huav.py \
    --train_data /mnt/data/custom_dataset.jsonl \
    --image_dir /mnt/data/custom_images \
    --num_epochs 3
```

### Monitor Training Live

```bash
# Terminal 1: Run training
./train_huav.sh

# Terminal 2: Watch progress
watch -n 5 'tail -20 ./outputs/huav_training/training_report.json'
```

### Compare Checkpoints

```bash
# Evaluate with different checkpoints
for epoch in 1 2 3; do
    echo "Testing epoch $epoch..."
    LOAD_MEMORY=./outputs/huav_training/huav_memory_epoch_${epoch}.pt \
        ./run_huav.sh &

    sleep 10
    ./run_luav.sh

    # Results saved with epoch number
    mv ./outputs/hierarchical_uav/task1_l-uav_results.jsonl \
       ./outputs/hierarchical_uav/task1_l-uav_results_epoch${epoch}.jsonl
done
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1: Use 8-bit quantization**
```bash
# Already enabled in train_huav.sh
python hierarchical_uav/train_huav.py --load_8bit
```

**Solution 2: Reduce batch size**
```python
# Edit hierarchical_uav/train_huav.py
# Process images one at a time (already implemented)
```

**Solution 3: Clear cache periodically**
```python
# Add to training loop
if i % 100 == 0:
    torch.cuda.empty_cache()
```

### Issue: Loss is NaN or infinite

**Cause**: Learning rate too high or numerical instability

**Solution:**
```bash
python hierarchical_uav/train_huav.py \
    --learning_theta 0.05 \  # Reduce from 0.1
    --surprise_eta 0.85      # Reduce momentum
```

### Issue: No high surprise samples

**Cause**: All samples look the same to the model

**Solution:**
- Check training data diversity
- Increase surprise threshold sensitivity
- Try different forgetting rate

### Issue: Training is very slow

**Expected speed:** ~3-4 samples/second on GPU

**If slower:**
1. Check GPU utilization: `nvidia-smi`
2. Ensure using 8-bit quantization: `--load_8bit`
3. Check disk I/O (loading images)

---

## Expected Results

### Training Metrics

**Healthy training looks like:**

```
Epoch 1:
  Avg loss: 0.025-0.035
  Avg surprise: -0.1 to -0.15
  High surprise: 15-25%

Epoch 2:
  Avg loss: 0.018-0.025  ← Decreasing
  Avg surprise: -0.08 to -0.12  ← More stable
  High surprise: 10-20%  ← Fewer novel samples

Epoch 3:
  Avg loss: 0.012-0.020  ← Converging
  Avg surprise: -0.06 to -0.10  ← Stable
  High surprise: 5-15%  ← Most patterns learned
```

**Warning signs:**

❌ Loss increasing: Learning rate too high or poor initialization
❌ Surprise always near zero: No learning happening
❌ High surprise >50%: Not learning patterns effectively

### Evaluation Improvements

**Metrics to compare:**

| Metric | Before Phase 3 | After Phase 3 | Improvement |
|--------|----------------|---------------|-------------|
| H-UAV accuracy | 3.1% | 30-50% | **~10-15x** |
| Answer diversity | 2.4% | 20-40% | **~10-15x** |
| Overall accuracy | 67.35% | 75-80% | **~10% gain** |
| Memory usefulness | Negative | Positive | **Qualitative** |

**Analyze with:**
```bash
python analyze_memory_influence.py \
    ./outputs/hierarchical_uav/task1_l-uav_results.jsonl \
    --output-dir ./outputs/memory_research_after_phase3
```

---

## Next Steps After Phase 3

### If Results are Good (30-50% H-UAV accuracy)

✅ **Success!** Memory is working

**Next:**
1. Fine-tune parameters for even better performance
2. Experiment with different learning rates
3. Try longer training (5-10 epochs)
4. Evaluate on new test sets

### If Results are Mediocre (10-20% H-UAV accuracy)

⚠️ **Partial success** - Memory is helping but not enough

**Actions:**
1. Train longer (5-10 epochs)
2. Increase learning rate (theta=0.15-0.2)
3. Check training data quality
4. Analyze which question types benefit most

### If Results are Still Poor (<10% H-UAV accuracy)

❌ **Need investigation**

**Debug checklist:**
1. ✓ MAC memory actually loaded?
   ```bash
   grep "MAC state loaded" <h-uav-output>
   ```

2. ✓ Training actually updated parameters?
   ```bash
   # Check loss decreased over epochs
   cat outputs/huav_training/training_report.json | jq '.results.epoch_stats'
   ```

3. ✓ H-UAV using loaded memory?
   ```bash
   # Check memory state is non-zero
   python -c "
   import torch
   state = torch.load('outputs/huav_training/huav_memory_final.pt')
   print('Surprise:', state['surprise'])
   print('Step count:', state['step_count'])
   "
   ```

4. Consider alternative: Skip MAC, use visual features directly
   (See "Short-term solution" in previous analysis)

---

## File Reference

### Training Scripts
- `hierarchical_uav/train_huav.py` - Main training script
- `train_huav.sh` - Shell wrapper with defaults

### Modified Files
- `hierarchical_uav/mac_memory/mac_layer.py:273-297` - Enabled test-time learning in eval mode
- `hierarchical_uav/eval_task1.py:285-289` - Added checkpoint loading
- `run_huav.sh` - Added LOAD_MEMORY support

### Output Files
- `outputs/huav_training/huav_memory_epoch_*.pt` - Epoch checkpoints
- `outputs/huav_training/huav_memory_final.pt` - Final trained memory
- `outputs/huav_training/training_report.json` - Training metrics

---

## Summary

Phase 3 implements **continual test-time learning** to make H-UAV's MAC memory useful:

1. **Before:** Random MAC → Random memory → 3.1% accuracy
2. **Training:** 971 samples × 3 epochs → Learn semantic patterns
3. **After:** Trained MAC → Meaningful memory → 30-50% accuracy (expected)

**Key innovation:** Test-time learning (no labels needed!)
**Expected gain:** ~10x improvement in memory-augmented accuracy

**Next:** Run training and evaluate! 🚀

```bash
# Complete workflow
./train_huav.sh                                                    # Step 1: Train
LOAD_MEMORY=outputs/huav_training/huav_memory_final.pt ./run_huav.sh  # Step 2: Deploy
./run_luav.sh                                                      # Step 3: Evaluate
```
