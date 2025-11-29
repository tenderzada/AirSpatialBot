# Hierarchical UAV System V2 - Quick Start Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Hierarchical UAV System V2                  │
│                (LoRA-based Memory Enhancement)               │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐                    ┌──────────────────┐
│   L-UAV (GPU 1)  │◄──── Socket ──────►│   H-UAV (GPU 0)  │
│                  │     localhost:50051 │                  │
│  LLaVA + LoRA    │                    │  LLaVA + LoRA    │
│   (3 epochs)     │                    │  (10 epochs)     │
│                  │                    │                  │
│ Basic Memory     │   Request Memory   │ Enhanced Memory  │
│  Capability      │ ─────────────────► │   Generation     │
│                  │                    │                  │
│                  │ ◄───────────────── │                  │
│   Inject Memory  │  Memory Tokens     │ LoRA-Enhanced    │
│   for Inference  │   [8, 4096]        │  Memory Tokens   │
└──────────────────┘                    └──────────────────┘
         │                                       │
         │                                       │
         ▼                                       ▼
  Lower confidence                       Higher confidence
  Basic predictions                     Enhanced predictions
```

## Key Design Principles

1. **H-UAV (High-UAV)**:
   - Uses **10-epoch LoRA** (best checkpoint)
   - Provides enhanced memory tokens to L-UAV
   - Higher computational capability
   - Better memory generation

2. **L-UAV (Low-UAV)**:
   - Uses **3-epoch LoRA** (final checkpoint)
   - Basic memory capability
   - Requests enhanced memory from H-UAV when needed
   - Injects H-UAV's memory for better inference

3. **Memory Injection**:
   - H-UAV generates memory tokens from 10-epoch LoRA
   - L-UAV receives and injects these tokens during inference
   - Hierarchical collaboration improves performance

## Setup Steps

### Step 1: Train LoRA Models

#### Train H-UAV LoRA (10 epochs)
```bash
./train_lora_injection_sqa.sh
```
- Output: `./outputs/lora_injection_sqa/lora_adapters_best.pt`
- Best checkpoint from 10 epochs

#### Train L-UAV LoRA (3 epochs)
```bash
./train_lora_injection_sqa_3epoch.sh
```
- Output: `./outputs/lora_injection_sqa_3epoch/lora_adapters_epochfinal.pt`
- Final checkpoint after 3 epochs

### Step 2: Start H-UAV Server

```bash
./start_huav_lora_socket_v2.sh
```

This starts the H-UAV memory server on `localhost:50051` with 10-epoch LoRA.

**Important**: H-UAV server must be running before L-UAV evaluation.

### Step 3: Run Experiments

#### Option A: Run All 3 Experiments
```bash
./run_collaboration_comparison_v2.sh
```

This runs:
1. **Experiment 1**: H-UAV Baseline (NO LoRA) - 200 samples
2. **Experiment 2**: H-UAV + MemGen (10-epoch LoRA) - 200 samples
3. **Experiment 3**: L-UAV (3-epoch) + H-UAV (10-epoch) - 200 samples

#### Option B: Run Individual Experiments

**Baseline (NO LoRA)**:
```bash
./eval_huav_baseline_200.sh
```

**H-UAV + MemGen (10-epoch LoRA)**:
```bash
./eval_huav_standalone_200.sh
```

**L-UAV + H-UAV Collaboration**:
```bash
./eval_luav_huav_collab_200_v2.sh
```

### Step 4: Compare Results

```bash
python compare_collaboration_results_3way_v2.py
```

## Expected Results

### Experiment 1: H-UAV Baseline (NO LoRA)
- Standard LLaVA performance
- No memory enhancement
- Establishes baseline

### Experiment 2: H-UAV + MemGen (10-epoch LoRA)
- **Expected**: Better than baseline
- LoRA memory enhancement active
- Shows benefit of LoRA training

### Experiment 3: L-UAV + H-UAV Collaboration
- **Expected**: L-UAV benefits from H-UAV's enhanced memory
- Demonstrates hierarchical collaboration
- Shows memory injection effectiveness

## File Structure

```
.
├── train_lora_injection_sqa.sh          # H-UAV training (10 epochs)
├── train_lora_injection_sqa_3epoch.sh   # L-UAV training (3 epochs)
│
├── start_huav_lora_socket_v2.sh         # Start H-UAV server
│
├── eval_huav_baseline_200.sh            # Exp 1: Baseline
├── eval_huav_standalone_200.sh          # Exp 2: H-UAV+MemGen
├── eval_luav_huav_collab_200_v2.sh      # Exp 3: L-UAV+H-UAV
│
├── run_collaboration_comparison_v2.sh   # Run all experiments
├── compare_collaboration_results_3way_v2.py  # Compare results
│
├── hierarchical_uav/
│   ├── train_lora_injection.py          # LoRA training script
│   ├── huav_lora_server_socket_v2.py    # H-UAV server (10-epoch LoRA)
│   ├── eval_sqa_lora_socket_v2.py       # L-UAV evaluation (3-epoch LoRA + memory injection)
│   ├── eval_huav_standalone.py          # H-UAV standalone evaluation
│   ├── eval_huav_baseline.py            # Baseline evaluation
│   └── mac_memory/
│       └── lora_injection.py            # LoRA injection implementation
│
└── outputs/
    ├── lora_injection_sqa/              # H-UAV LoRA (10 epochs)
    │   └── lora_adapters_best.pt
    ├── lora_injection_sqa_3epoch/       # L-UAV LoRA (3 epochs)
    │   └── lora_adapters_epochfinal.pt
    ├── comparison_huav_baseline_200/    # Exp 1 results
    ├── comparison_huav_standalone_200/  # Exp 2 results
    └── comparison_luav_huav_collab_200_v2/  # Exp 3 results
```

## Troubleshooting

### Problem: H-UAV server connection fails

**Solution**:
```bash
# Check if H-UAV server is running
netstat -an | grep 50051

# If not, start it:
./start_huav_lora_socket_v2.sh
```

### Problem: LoRA weights not found

**Solution**:
```bash
# For H-UAV (10 epochs):
./train_lora_injection_sqa.sh

# For L-UAV (3 epochs):
./train_lora_injection_sqa_3epoch.sh
```

### Problem: GPU out of memory

**Solution**:
- Reduce batch size in training scripts
- Use `--load_8bit` flag (already enabled)
- Ensure only one model per GPU (H-UAV on GPU 0, L-UAV on GPU 1)

### Problem: Results are identical (no improvement)

**Check**:
1. LoRA weights are actually loaded (check logs)
2. H-UAV server is using 10-epoch LoRA
3. L-UAV is receiving memory tokens from H-UAV
4. Memory injection is working (check logs for "💉 Injecting H-UAV's enhanced memory")

## Performance Metrics

The comparison tool shows:

1. **Overall Metrics**: MAE, RMSE, MRE
2. **Performance Gains**: % improvement over baseline
3. **Collaboration Stats**: H-UAV usage rate, request counts
4. **Per-Type Metrics**: Performance breakdown by question type
5. **Ranking**: Best performing approach

## Key Differences from V1

| Feature | V1 | V2 |
|---------|----|----|
| L-UAV | Vanilla LLaVA | LLaVA + 3-epoch LoRA |
| H-UAV | LLaVA + LoRA | LLaVA + 10-epoch LoRA |
| Memory Injection | ❌ Not implemented | ✅ Implemented |
| Expected Improvement | ❌ None (bug) | ✅ Hierarchical benefit |

## Next Steps

After running experiments:

1. **Analyze results** using the comparison tool
2. **Tune hyperparameters** if needed:
   - LoRA rank
   - Number of epochs
   - Confidence threshold
   - Force query frequency

3. **Scale up** to full dataset if results are promising
4. **Experiment** with different architectures:
   - More LoRA injection layers
   - Different memory token counts
   - Alternative injection strategies

## Citations & References

- **LLaVA**: Liu et al., "Visual Instruction Tuning"
- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
- **Memory Weaver**: Custom architecture for hierarchical UAV collaboration
