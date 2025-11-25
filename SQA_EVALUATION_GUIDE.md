# SQA Evaluation Guide for Hierarchical UAV System

## Overview

This guide explains how to evaluate the Hierarchical UAV (H-UAV + L-UAV) system on the **SQA (Spatial Question Answering)** task. SQA is a spatial regression task where the model predicts numeric spatial attributes of vehicles in aerial images.

### Why SQA Task?

The SQA task is **ideal for evaluating high-low UAV collaboration** because:

✅ **Single VLM Task**: Only requires visual understanding + spatial sensing (no multi-step planning)
✅ **Pure Spatial Regression**: Tests spatial awareness directly with numeric outputs
✅ **Large Test Set**: 17,526 samples for comprehensive evaluation
✅ **Clear Metrics**: RMSE, MAE, R² provide objective performance measures
✅ **6 Question Types**: Covers diverse spatial queries (depth, distance, size, etc.)

**Contrast with Task1**: Task1 is a multi-agent task requiring LLM planning + VLM + knowledge base lookup + summarization, making it harder to isolate the contribution of memory augmentation.

---

## Dataset: airspatial_sqa_test.jsonl

**Total Samples**: 17,526 (pure test set)
**File Location**: `./data/metadata/airspatial_sqa_test.jsonl`

### Data Structure

```json
{
  "question_id": 0,
  "qtype": "depth",
  "image_id": "0a2c4f75-DJI_0001.JPG",
  "bbox": [2322, 870, 2428, 988],
  "bbox_3d": [-22.5, -43.5, 30.0, 5.8, 3.8, 1.9, -12.8],
  "question": "Please tell me the depth of the car<bbox>[[2322, 870, 2428, 988]]</bbox> in the image. (Unit: meter)",
  "ground_truth": 40.0,
  "dataset": "airspatial"
}
```

### Question Types (6 types, ~2,921 samples each)

| Type | Description | Unit | Example Question |
|------|-------------|------|------------------|
| **depth** | Depth of vehicle | meters | "What is the depth of the car...?" |
| **distance** | Distance from camera/drone | meters | "How many meters is the car from the drone?" |
| **length** | Vehicle length | millimeters | "What is the length of the car...?" |
| **width** | Vehicle width | millimeters | "What is the width of the car...?" |
| **height** | Vehicle height | millimeters | "How high is the car...?" |
| **size** | All dimensions (L/W/H) | millimeters | "Output the length, width, and height..." |

### Key Features

- ✅ **Complete questions**: Natural language questions with context
- ✅ **Bbox in question**: `<bbox>[[x,y,x,y]]</bbox>` specifies target vehicle
- ✅ **Numeric answers**: Ground truth is a numeric value (meters or millimeters)
- 📊 **Regression metrics**: RMSE, MAE, R-squared for evaluation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SQA Evaluation Flow                       │
└─────────────────────────────────────────────────────────────┘

 [Image + Question]
        │
        ├──> Parse bbox from question
        ├──> Crop image to bbox region (focused spatial analysis)
        ├──> Extract visual features
        │
        v
 ┌──────────────┐
 │   L-UAV      │ (Lightweight Agent)
 │   - LLaVA    │
 │   - Self-    │ Compute self-match score
 │     Matching │ Decision: query H-UAV?
 └──────┬───────┘
        │
        ├─── High confidence ──> Local inference only
        │
        └─── Low confidence ──> Query H-UAV for memory
                                       │
                                       v
                              ┌──────────────┐
                              │   H-UAV      │ (Resource-Rich Agent)
                              │   - LLaVA    │
                              │   - MAC Mem. │ Trained spatial knowledge
                              └──────┬───────┘
                                     │
                                     v
                              Memory features
                                     │
        ┌────────────────────────────┘
        │
        v
 Memory-augmented generation
        │
        v
 [Numeric Answer]
        │
        v
 Evaluate: RMSE, MAE, R²
```

---

## Quick Start

### Prerequisites

1. **Environment Setup**: Follow `SETUP_ENVIRONMENT.md`
2. **Data**: Download SQA test set to `./data/metadata/airspatial_sqa_test.jsonl`
3. **Images**: Ensure images are in `./data/images/`
4. **Model**: LLaVA model in `./models/AirSpatialBot/`
5. **CLIP Vision**: CLIP model in `/mnt/data/clip-vit-large-patch14-336`

### Method 1: Using the Run Script (Recommended)

Make the script executable:
```bash
chmod +x run_sqa_evaluation.sh
```

#### Start H-UAV Server
```bash
./run_sqa_evaluation.sh h-uav
```

#### Start L-UAV Client (in another terminal)
```bash
./run_sqa_evaluation.sh l-uav
```

#### Run Both Automatically (requires tmux)
```bash
./run_sqa_evaluation.sh both
```

#### Run L-UAV Standalone (Baseline)
```bash
./run_sqa_evaluation.sh l-uav --standalone
```

#### Test with 100 Samples
```bash
./run_sqa_evaluation.sh l-uav --max-samples 100
```

### Method 2: Manual Execution

#### H-UAV Server
```bash
python hierarchical_uav/eval_sqa.py \
    --uav_type h-uav \
    --device cuda:0 \
    --port 50051 \
    --model_path ./models/AirSpatialBot \
    --vision_tower /mnt/data/clip-vit-large-patch14-336 \
    --test_data ./data/metadata/airspatial_sqa_test.jsonl \
    --image_dir ./data/images \
    --output ./outputs/hierarchical_uav/sqa/sqa_huav_results.jsonl \
    --load_8bit
```

#### L-UAV Client
```bash
python hierarchical_uav/eval_sqa.py \
    --uav_type l-uav \
    --device cuda:1 \
    --huav_address localhost:50051 \
    --threshold 0.7 \
    --model_path ./models/AirSpatialBot \
    --vision_tower /mnt/data/clip-vit-large-patch14-336 \
    --test_data ./data/metadata/airspatial_sqa_test.jsonl \
    --image_dir ./data/images \
    --output ./outputs/hierarchical_uav/sqa/sqa_luav_results.jsonl \
    --load_8bit
```

#### L-UAV Standalone (Baseline)
```bash
python hierarchical_uav/eval_sqa.py \
    --uav_type l-uav \
    --device cuda:0 \
    --model_path ./models/AirSpatialBot \
    --vision_tower /mnt/data/clip-vit-large-patch14-336 \
    --test_data ./data/metadata/airspatial_sqa_test.jsonl \
    --image_dir ./data/images \
    --output ./outputs/hierarchical_uav/sqa/sqa_luav_standalone_results.jsonl \
    --load_8bit \
    --standalone
```

---

## Evaluation Metrics

### Regression Metrics

The evaluation script computes three standard regression metrics:

1. **MAE (Mean Absolute Error)**
   - Average absolute difference between predictions and ground truth
   - Lower is better
   - Unit: same as ground truth (meters or millimeters)
   - Formula: `MAE = (1/n) * Σ|y_pred - y_true|`

2. **RMSE (Root Mean Square Error)**
   - Square root of average squared errors
   - Penalizes large errors more heavily
   - Lower is better
   - Formula: `RMSE = sqrt((1/n) * Σ(y_pred - y_true)²)`

3. **R² (R-squared / Coefficient of Determination)**
   - Proportion of variance explained by the model
   - Range: (-∞, 1], where 1 is perfect prediction
   - Higher is better
   - Formula: `R² = 1 - (SS_res / SS_tot)`

### Metrics Breakdown

The evaluation provides metrics for:
- **Overall**: Across all samples
- **By Question Type**: Separate metrics for depth, distance, length, width, height, size
- **By Source**: Compare local vs. H-UAV-augmented performance

---

## Output Format

Results are saved as JSONL files with the following structure:

```json
{
  "question_id": 0,
  "image_id": "0a2c4f75-DJI_0001.JPG",
  "question": "Please tell me the depth of the car<bbox>[[2322, 870, 2428, 988]]</bbox>...",
  "qtype": "depth",
  "answer_text": "The depth of the car is approximately 42.5 meters.",
  "predicted_value": 42.5,
  "ground_truth": 40.0,
  "self_match_score": 0.6234,
  "source": "huav",
  "cache_hit": false
}
```

### Fields

- `question_id`: Unique identifier for the question
- `image_id`: Image filename
- `question`: Full question text with bbox
- `qtype`: Question type (depth, distance, etc.)
- `answer_text`: Raw text output from model
- `predicted_value`: Extracted numeric prediction
- `ground_truth`: True numeric value
- `self_match_score`: Self-matching confidence score
- `source`: Decision source (`local`, `huav`, `local_fallback`)
- `cache_hit`: Whether H-UAV cache was used

---

## Experimental Comparisons

### Baseline vs. Memory-Augmented

To evaluate the impact of H-UAV memory augmentation:

1. **Baseline (L-UAV Standalone)**:
   ```bash
   ./run_sqa_evaluation.sh l-uav --standalone
   ```

2. **Memory-Augmented (L-UAV + H-UAV)**:
   ```bash
   # Terminal 1
   ./run_sqa_evaluation.sh h-uav

   # Terminal 2
   ./run_sqa_evaluation.sh l-uav
   ```

3. **Compare Results**:
   - Check RMSE/MAE/R² improvements
   - Analyze by question type
   - Examine self-match score distribution

### With Trained MAC Memory

If you have trained MAC memory (from Phase 3):

```bash
# H-UAV with trained memory
python hierarchical_uav/eval_sqa.py \
    --uav_type h-uav \
    --device cuda:0 \
    --port 50051 \
    --load-memory ./outputs/huav_training/huav_memory_final.pt \
    ...
```

---

## Performance Optimization

### Memory Optimization

- **8-bit Quantization** (default): `--load_8bit`
  - Reduces GPU memory by ~50%
  - Minimal accuracy loss
  - Recommended for RTX 4090 (24GB)

- **4-bit Quantization** (experimental): `--load_4bit`
  - Further reduces memory
  - May impact accuracy

### Speed Optimization

- **Adjust max_new_tokens**: SQA only needs short numeric answers
  - Default: 64 tokens (vs. 512 for Task1)
  - Faster generation

- **Use do_sample=False**: Greedy decoding for deterministic results

---

## Troubleshooting

### H-UAV Connection Issues

**Problem**: L-UAV cannot connect to H-UAV

**Solutions**:
1. Ensure H-UAV is started first
2. Check port: `netstat -tuln | grep 50051`
3. Verify H-UAV logs show "✓ H-UAV Server started"
4. Try ping: the script checks connectivity automatically

### Out of Memory

**Problem**: CUDA out of memory

**Solutions**:
1. Use `--load_8bit` (already default)
2. Reduce batch size (eval uses batch=1)
3. Use smaller model or different GPU
4. Test with `--max-samples 100` first

### Invalid Predictions

**Problem**: Model outputs non-numeric answers

**Solutions**:
1. Check `predicted_value` field (should be numeric)
2. Review `answer_text` for parsing issues
3. Script logs "Invalid predictions: X%" in summary
4. May need better prompting for specific question types

### Image Not Found

**Problem**: Images missing

**Solutions**:
1. Verify image directory: `--image_dir ./data/images`
2. Check image_id matches filenames
3. Script skips missing images with warning

---

## Key Implementation Features

### 1. Bbox Cropping
- Crops image to bbox region with 15% padding
- Focuses model attention on target vehicle
- Critical for accurate spatial predictions

### 2. Memory Injection
- H-UAV memory features augmented with dynamic weighting
- Weight based on self-match score (0.1 - 0.9)
- Higher confidence → higher weight

### 3. Self-Matching Module
- Computes similarity between current sample and H-UAV's memory
- Threshold (default 0.7) determines query decision
- Tracks statistics: query rate, score distribution

### 4. Numeric Extraction
- Regex-based extraction from text output
- Handles various formats (e.g., "42.5 meters", "4500 mm")
- Falls back gracefully for parsing failures

---

## Expected Results

### Untrained Memory (Random Initialization)

With random MAC memory (no training):
- **Self-match scores**: ~0.5 (random baseline)
- **Query rate**: ~50% (threshold-dependent)
- **Performance**: Similar to standalone (no memory benefit)

### Trained Memory (After Phase 3)

With trained MAC memory:
- **Self-match scores**: More discriminative (0.2 - 0.9 range)
- **Query rate**: Adaptive based on difficulty
- **Performance**:
  - Lower RMSE/MAE on challenging samples
  - Higher R² overall
  - Bigger improvement on distance/depth (harder types)

---

## Next Steps

After SQA evaluation:

1. **Analyze Results**:
   - Compare baseline vs. memory-augmented
   - Identify which question types benefit most
   - Examine error patterns

2. **Train MAC Memory** (if not done):
   - Follow Phase 3 training guide
   - Re-evaluate with trained memory

3. **Optimize Hyperparameters**:
   - Tune self-match threshold
   - Adjust memory weight range
   - Experiment with different prompts

4. **Scale Up**:
   - Evaluate on full 17,526 samples
   - Analyze performance vs. query rate trade-off

---

## References

- **eval_sqa.py**: Main evaluation script
- **run_sqa_evaluation.sh**: Convenient runner script
- **HIERARCHICAL_UAV_README.md**: System architecture overview
- **MEMORY_INJECTION_IMPLEMENTATION.md**: Memory injection details

---

## Questions?

For issues or questions:
1. Check script logs for detailed error messages
2. Review troubleshooting section above
3. Verify prerequisites are met
4. Test with `--max-samples 100` for quick debugging
