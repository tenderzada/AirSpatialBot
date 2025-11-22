# Force Query Rate - Research Mode Guide

## Purpose

This feature allows you to control what percentage of samples query H-UAV for memory, enabling better testing of the memory mechanism's effectiveness.

## Why This is Needed

**Current Behavior:**
- Knowledge base answers ~96.5% of questions immediately
- Only ~3.5% of samples reach self-matching and potentially query H-UAV
- This gives insufficient data to evaluate memory mechanism

**Research Goal:**
- Test whether memory from H-UAV actually enhances L-UAV inference
- Analyze memory's influence on reasoning (not just accuracy)
- Gather sufficient samples to understand when memory helps/hurts

## How It Works

### Normal Mode (default)
```
Sample → KB lookup → Found? → Return (no memory)
                   → Not found → Self-matching → Query H-UAV
```
Result: ~3.5% query rate

### Research Mode (force-query-rate > 0)
```
Sample → Random(force_rate)? → Yes → Skip KB → Self-matching → Query H-UAV
                             → No → Normal flow (KB → self-matching)
```
Result: Controllable query rate

## Usage

### Method 1: Edit Shell Script (Recommended)

Edit `run_luav.sh`, change line 25:
```bash
# From:
FORCE_QUERY_RATE=0.0

# To (example - force 30% to query):
FORCE_QUERY_RATE=0.3
```

Then run normally:
```bash
./run_luav.sh
```

### Method 2: Command Line

```bash
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device cuda:1 \
    --huav_address localhost:50051 \
    --force-query-rate 0.3 \
    ...other params...
```

## Recommended Settings

### For Memory Research

| Goal | Recommended Rate | Expected Queries |
|------|------------------|------------------|
| Quick test | 0.1 (10%) | ~100 samples |
| Balanced test | 0.2-0.3 (20-30%) | ~200-300 samples |
| Thorough test | 0.5 (50%) | ~500 samples |
| Full test | 1.0 (100%) | ~1000 samples |

**Recommended starting point:** `0.3` (30%)
- Provides ~300 samples for analysis
- Balances accuracy (70% still use KB) with research needs
- Sufficient to detect memory patterns

### For Production Use

```bash
FORCE_QUERY_RATE=0.0  # Disabled - use normal self-matching
```

## What You'll See

### Startup Message (if rate > 0)
```
⚠️  RESEARCH MODE: Force query rate = 30.0%
   → 30.0% of samples will bypass KB and test H-UAV memory
```

### Statistics Output
```
============================================================
L-UAV Evaluation Statistics
============================================================
Total samples: 971
Local decisions: 650 (67.0%)
Remote queries: 321 (33.0%)
  ↳ Forced queries (research mode): 291 (30.0%)
  ↳ Natural queries (self-matching): 30 (3.1%)
```

This shows:
- **Total queries**: 33% (up from 3.5%)
- **Forced**: 291 samples bypassed KB (30% as configured)
- **Natural**: 30 samples queried via self-matching (normal 3.5%)

## Analysis After Running

Use the analysis tool to study memory influence:

```bash
python analyze_memory_influence.py \
    ./outputs/hierarchical_uav/task1_l-uav_results.jsonl \
    --output-dir ./outputs/memory_research
```

This will show:
- How many samples used memory (should match forced + natural queries)
- Answer diversity (unique outputs with vs without memory)
- Self-match score distribution
- Evidence of memory's influence

## Example Experiment

**Test if memory improves inference for specific question types:**

1. **Run with 30% forced queries:**
```bash
# Edit run_luav.sh: FORCE_QUERY_RATE=0.3
./run_luav.sh
```

2. **Analyze results:**
```bash
python analyze_memory_influence.py \
    ./outputs/hierarchical_uav/task1_l-uav_results.jsonl
```

3. **Check findings:**
- Do memory-augmented samples produce different answers?
- Is there a pattern to which samples benefit?
- Does self-match score correlate with answer quality?

4. **Compare with ablation study:**
```bash
# Test different memory weights
python memory_ablation_study.py \
    ./outputs/hierarchical_uav/task1_l-uav_results.jsonl
```

## Important Notes

### This is a Research Tool
- **Purpose**: Test memory mechanism effectiveness
- **Not for production**: Bypassing KB reduces overall accuracy
- **For analysis**: Generates data to understand memory's role

### When to Use
✅ Researching memory mechanism
✅ Testing Phase 1/2/3 optimizations
✅ Analyzing memory influence patterns
✅ Gathering data for ablation studies

❌ Production evaluation (use 0.0)
❌ Reporting final accuracy metrics
❌ When KB answers are known to be correct

### Effect on Accuracy
- **Overall accuracy**: Will likely decrease (KB is 100% accurate)
- **Cache-hit accuracy**: More samples to evaluate memory quality
- **Research value**: Increased (more data about memory mechanism)

**Trade-off**: Lower overall accuracy, but better understanding of memory

## Troubleshooting

### All samples still use KB
**Issue**: `forced_query_count = 0` even with rate > 0

**Solution**:
- Check you're running L-UAV, not H-UAV
- Verify parameter is passed: `--force-query-rate 0.3`
- Check config.force_query_rate is set correctly

### Query rate lower than expected
**Example**: Set 0.3 but only get 20%

**Explanation**:
- Rate applies to total samples
- Some may still fail due to errors or missing data
- Actual rate ≈ configured rate × (1 - error_rate)

### Too many queries (slow)
**Issue**: Running with rate 1.0 is very slow

**Solution**:
- Start with 0.3 (30%) for balanced speed/data
- Only use 1.0 for comprehensive analysis
- Consider reducing test set size

## Next Steps

After gathering forced query data:

1. **Analyze memory influence** (see MEMORY_RESEARCH_GUIDE.md)
2. **Run ablation studies** to test different memory weights
3. **Identify patterns** in when memory helps vs hurts
4. **Optimize** based on findings (Phase 2/3)

---

**Quick Start for Memory Research:**

```bash
# 1. Edit run_luav.sh
#    Change: FORCE_QUERY_RATE=0.3

# 2. Run evaluation
./run_luav.sh

# 3. Analyze results
python analyze_memory_influence.py \
    ./outputs/hierarchical_uav/task1_l-uav_results.jsonl
```

This will give you ~300 samples to analyze memory mechanism effectiveness!
