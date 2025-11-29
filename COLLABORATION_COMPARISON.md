# 协作对比实验：H-UAV vs L-UAV+H-UAV (200 样本)

## 实验目的

在前 200 个样本上对比两种模式的性能：
1. **H-UAV Standalone** - 使用训练好的 LoRA MemGen (10 epochs 最佳)
2. **L-UAV + H-UAV Collaboration** - L-UAV 利用 H-UAV 的 memory 增强推理

## 实验设计

### 实验 1: H-UAV Standalone with LoRA MemGen

**配置**:
- 模型: LLaVA + LoRA @ layers 8,16,24
- LoRA 权重: `lora_adapters_best.pt` (10 epochs 训练最佳)
- 样本数: 前 200 个
- 设备: cuda:0

**预期**:
- 高性能基线
- LoRA 增强的 memory generation
- 直接推理，无协作开销

### 实验 2: L-UAV + H-UAV Collaboration

**配置**:
- L-UAV: 基础 LLaVA (无 LoRA)
- H-UAV: LLaVA + LoRA MemGen
- L-UAV 设备: cuda:1
- H-UAV 地址: localhost:50051
- **强制 query**: 每 5 个样本
- **自适应 query**: confidence < 0.5
- 样本数: 前 200 个

**预期**:
- L-UAV 从 H-UAV memory 中受益
- 协作机制有效性验证
- H-UAV 使用率 ~20-40%

## 使用方法

### 准备工作

1. **确保 LoRA 训练完成**:
```bash
# 应该存在以下文件
ls ./outputs/lora_injection_sqa/lora_adapters_best.pt
```

2. **启动 H-UAV Server**:
```bash
./start_huav_lora_socket.sh
```

### 方式 1: 一键运行完整实验（推荐）

```bash
./run_collaboration_comparison.sh
```

**流程**:
1. 检查 H-UAV server 连接
2. 运行实验 1: H-UAV Standalone (200 samples)
3. 运行实验 2: L-UAV + H-UAV Collab (200 samples)
4. 生成对比报告

**预期时间**: ~30-40 分钟

### 方式 2: 分步运行

#### 步骤 1: H-UAV Standalone

```bash
./eval_huav_standalone_200.sh
```

**输出**:
```
============================================================
Comparison Experiment 1: H-UAV Standalone with MemGen LoRA
============================================================

🧪 Comparison Experiment Configuration:
  Condition: H-UAV Standalone with LoRA MemGen
  Samples: First 200 samples
  LoRA weights: lora_adapters_best.pt (best from 10 epochs)
  ...

H-UAV Standalone Evaluation: 100%|██████████| 200/200

✅ H-UAV Standalone Evaluation Complete

📊 Results (H-UAV Standalone with MemGen):
  MAE:  XXX.XX
  RMSE: XXX.XX
  MRE:  XX.XX%
```

#### 步骤 2: L-UAV + H-UAV Collaboration

```bash
./eval_luav_huav_collab_200.sh
```

**输出**:
```
============================================================
Comparison Experiment 2: L-UAV + H-UAV Collaboration
============================================================

🧪 Comparison Experiment Configuration:
  Condition: L-UAV + H-UAV Collaboration
  Samples: First 200 samples
  Force query: Every 5 samples
  ...

L-UAV + H-UAV Evaluation: 100%|██████████| 200/200

✅ L-UAV + H-UAV Collaboration Evaluation Complete

📊 Results (L-UAV + H-UAV Collaboration):
  MAE:  XXX.XX
  RMSE: XXX.XX
  MRE:  XX.XX%

Collaboration Stats:
  H-UAV usage rate: XX.X%
```

#### 步骤 3: 对比分析

```bash
python compare_collaboration_results.py
```

**输出示例**:
```
==========================================================================================
                Collaboration Comparison: H-UAV vs L-UAV+H-UAV (200 samples)
==========================================================================================

📊 Overall Metrics Comparison:
==========================================================================================
Metric          H-UAV Standalone          L-UAV+H-UAV               Difference
------------------------------------------------------------------------------------------
MAE             115.23                    108.45                    -6.78 (-5.9%)
RMSE            205.12                    195.67                    -9.45 (-4.6%)
MRE             25.34%                    23.12%                    -2.22 pp (-8.8%)
==========================================================================================

🤝 Collaboration Statistics:
------------------------------------------------------------------------------------------
  H-UAV requests:        40
  Standalone inferences: 160
  H-UAV usage rate:      20.0%

📋 Per-Type Metrics (MRE) Comparison:
==========================================================================================
Question Type   H-UAV Standalone          L-UAV+H-UAV               Difference
------------------------------------------------------------------------------------------
depth           89.45%                    85.23%                    -4.22 pp (-4.7%)
distance        15.67%                    14.34%                    -1.33 pp (-8.5%)
height          12.34%                    11.23%                    -1.11 pp (-9.0%)
length          8.91%                     8.45%                     -0.46 pp (-5.2%)
size            24.56%                    22.90%                    -1.66 pp (-6.8%)
width           3.21%                     3.12%                     -0.09 pp (-2.8%)
==========================================================================================

💡 Analysis:
------------------------------------------------------------------------------------------
✅ L-UAV+H-UAV Collaboration OUTPERFORMS H-UAV Standalone:
   - MAE improved by 5.9%
   - MRE improved by 2.22 percentage points

   Interpretation:
   - L-UAV benefits from H-UAV's LoRA-enhanced memory
   - Collaboration mechanism is effective
   - Forced queries every 5 samples help guide L-UAV

🔑 Key Findings:
------------------------------------------------------------------------------------------
1. H-UAV Standalone (with LoRA MemGen):
   - MAE: 115.23, MRE: 25.34%
   - Strong baseline with LoRA-enhanced memory

2. L-UAV + H-UAV Collaboration:
   - MAE: 108.45, MRE: 23.12%
   - H-UAV usage rate: 20.0%
   - Demonstrates collaboration mechanism

3. Comparison:
   ✅ Collaboration improves performance by 5.9%
==========================================================================================
```

## 结果解读

### 关键指标

- **MAE** (Mean Absolute Error): 平均绝对误差，越低越好
- **RMSE** (Root Mean Square Error): 均方根误差，越低越好
- **MRE** (Mean Relative Error): 平均相对误差，越低越好
- **H-UAV Usage Rate**: H-UAV 被调用的比例

### 预期结果

#### 情况 1: 协作优于单独（理想）

```
✅ L-UAV+H-UAV Collaboration OUTPERFORMS H-UAV Standalone
   - MAE improved by X%
   - MRE improved by X percentage points
```

**解释**:
- L-UAV 成功利用了 H-UAV 的 LoRA-enhanced memory
- 协作机制有效
- Memory 引导帮助 L-UAV 做出更好的预测

#### 情况 2: 单独优于协作

```
⚠️  H-UAV Standalone OUTPERFORMS L-UAV+H-UAV Collaboration
   - MAE worse by X%
```

**可能原因**:
- Memory 集成尚未充分有效
- L-UAV 可能需要更好的 memory token 集成方式
- 考虑调整 confidence threshold 或 query 频率

#### 情况 3: 性能相近

```
➡️  Similar Performance
   - MAE difference: ±2%
```

**解释**:
- 两种方法效果相当
- L-UAV+H-UAV 展示了潜力，但需要进一步优化

## 输出文件

### 结果文件

```
./outputs/comparison_huav_standalone_200/
└── results.json
    ├── metrics: {mae, rmse, mre}
    ├── type_metrics: {...}
    └── results: [...]

./outputs/comparison_luav_huav_collab_200/
└── results.json
    ├── metrics: {mae, rmse, mre}
    ├── collaboration_stats: {huav_requests, ...}
    ├── type_metrics: {...}
    └── results: [...]
```

### 查看详细结果

```bash
# H-UAV standalone 详细结果
cat ./outputs/comparison_huav_standalone_200/results.json | jq .

# L-UAV+H-UAV collaboration 详细结果
cat ./outputs/comparison_luav_huav_collab_200/results.json | jq .

# 对比指标
jq -s '{
  "huav": .[0].metrics,
  "collab": .[1].metrics
}' \
  ./outputs/comparison_huav_standalone_200/results.json \
  ./outputs/comparison_luav_huav_collab_200/results.json
```

## 故障排除

### H-UAV server 未运行

```bash
❌ H-UAV server not reachable at localhost:50051

解决: 启动 H-UAV server
./start_huav_lora_socket.sh
```

### LoRA 权重未找到

```bash
❌ LoRA weights not found: lora_adapters_best.pt

解决: 先训练 LoRA
./train_lora_injection_sqa.sh
```

### CUDA OOM

```bash
解决:
1. 减少 batch size
2. 使用 8-bit 量化 (已默认启用)
3. 确保 H-UAV 和 L-UAV 在不同 GPU 上
```

## 实验变体

### 调整强制 query 频率

编辑 `eval_luav_huav_collab_200.sh`:
```bash
FORCE_HUAV_EVERY_N=5  # 改为 10, 20 等
```

### 调整 confidence threshold

编辑 `eval_luav_huav_collab_200.sh`:
```bash
CONFIDENCE_THRESHOLD=0.5  # 改为 0.3, 0.7 等
```

### 增加样本数

编辑两个评估脚本:
```bash
MAX_SAMPLES=200  # 改为 500, 1000 等
```

## 后续步骤

完成对比实验后：

1. **分析结果** - 理解协作机制的效果
2. **优化策略** - 调整 query 策略或 memory 集成
3. **扩大实验** - 在全部 17,526 样本上评估
4. **论文撰写** - 使用实验结果支持论文

## 快速参考

```bash
# 完整实验套件
./run_collaboration_comparison.sh

# 单独运行
./eval_huav_standalone_200.sh      # 实验 1
./eval_luav_huav_collab_200.sh     # 实验 2
python compare_collaboration_results.py  # 对比

# 查看 H-UAV server 状态
ps aux | grep huav_lora_server

# 停止 H-UAV server
pkill -f huav_lora_server
```

---

**Created**: 2025-11-28
**Branch**: `claude/memgen-lora-integration-01Hy2BbsKDVozpNVZkpaGtWU`
**Purpose**: Comprehensive collaboration comparison on 200 samples
