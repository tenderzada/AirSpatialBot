# H-UAV 消融实验：LoRA Memory Weaver 效果验证

## 实验目的

验证 LoRA Memory Weaver 对 H-UAV 性能的提升效果。

通过对比**有 LoRA** 和**无 LoRA** (baseline) 两种条件下的性能，量化 LoRA memory weaver 的贡献。

## 实验设计

### 对比条件

| 条件 | 模型 | 描述 |
|------|------|------|
| **WITH LoRA** | LLaVA + LoRA @ layers 8,16,24 | LoRA 增强的记忆编织器 |
| **WITHOUT LoRA** | LLaVA (baseline) | 基础 LLaVA，无记忆增强 |

### 评估配置

- **样本数**: 前 100 个样本（减少成本）
- **数据集**: AirSpatial SQA 测试集
- **设备**: cuda:0
- **量化**: 8-bit
- **评估指标**: MAE, RMSE, MRE (按问题类型)

## 使用方法

### 步骤 1: 运行 WITH LoRA 评估

```bash
./eval_huav_ablation_with_lora.sh
```

**输出**:
- 结果保存到: `./outputs/ablation_huav_with_lora/results.json`
- 评估 100 个样本
- 加载训练好的 LoRA 权重

### 步骤 2: 运行 WITHOUT LoRA 评估

```bash
./eval_huav_ablation_without_lora.sh
```

**输出**:
- 结果保存到: `./outputs/ablation_huav_without_lora/results.json`
- 评估相同的 100 个样本
- 使用基础 LLaVA（无 LoRA）

### 步骤 3: 对比结果

```bash
python compare_ablation_results.py
```

**输出示例**:
```
================================================================================
                    H-UAV Ablation Study: LoRA Memory Weaver
================================================================================

📊 Overall Metrics Comparison (100 samples):
================================================================================
Metric          WITH LoRA            WITHOUT LoRA         Improvement
--------------------------------------------------------------------------------
MAE             115.23               142.56               -19.14%
RMSE            205.12               248.37               -17.41%
MRE             25.34%               31.89%               -6.55 pp
================================================================================

📋 Per-Type Metrics (MRE):
================================================================================
Question Type   WITH LoRA            WITHOUT LoRA         Improvement
--------------------------------------------------------------------------------
depth           89.45%               95.23%               -5.78 pp
distance        15.67%               19.34%               -3.67 pp
height          12.34%               15.89%               -3.55 pp
length          8.91%                11.23%               -2.32 pp
size            24.56%               28.90%               -4.34 pp
width           3.21%                4.56%                -1.35 pp
================================================================================

💡 Summary:
--------------------------------------------------------------------------------
✅ LoRA Memory Weaver IMPROVES performance:
   - MAE reduced by 19.14%
   - MRE reduced by 6.55 percentage points

Note: Lower MAE/RMSE/MRE is better
      Positive improvement means WITH LoRA performs better
================================================================================
```

## 结果解读

### 关键指标

- **MAE** (Mean Absolute Error): 平均绝对误差，越低越好
- **RMSE** (Root Mean Square Error): 均方根误差，越低越好
- **MRE** (Mean Relative Error): 平均相对误差，越低越好

### 预期结果

如果 LoRA Memory Weaver 有效：
- ✅ WITH LoRA 的 MAE/RMSE/MRE 应该**低于** WITHOUT LoRA
- ✅ Improvement 应该为**负值**（表示误差减少）
- ✅ 在困难任务（如 depth）上提升更明显

### Per-Type 分析

查看不同问题类型的性能：
- **Depth**: 最具挑战性，期望 LoRA 提升最大
- **Width**: 相对简单，期望提升较小
- **Size/Length/Height**: 中等难度

## 文件说明

### 评估脚本

1. **`eval_huav_ablation_with_lora.sh`**
   - 评估 H-UAV WITH LoRA
   - 调用 `hierarchical_uav/eval_huav_standalone.py`
   - 加载 LoRA 权重

2. **`eval_huav_ablation_without_lora.sh`**
   - 评估 H-UAV WITHOUT LoRA
   - 调用 `hierarchical_uav/eval_huav_baseline.py`
   - 使用基础 LLaVA

### Python 模块

1. **`hierarchical_uav/eval_huav_standalone.py`**
   - H-UAV with LoRA evaluation
   - LoRA injection at layers 8, 16, 24
   - Memory-enhanced inference

2. **`hierarchical_uav/eval_huav_baseline.py`**
   - H-UAV baseline evaluation
   - Vanilla LLaVA (no LoRA)
   - Same evaluation pipeline

### 工具脚本

1. **`compare_ablation_results.py`**
   - 自动对比工具
   - 读取两个 results.json
   - 计算改进百分比
   - 生成格式化报告

## 输出文件

### WITH LoRA
```
./outputs/ablation_huav_with_lora/
└── results.json
    ├── metrics: {mae, rmse, mre}
    ├── type_metrics: {depth, distance, ...}
    └── results: [...详细结果...]
```

### WITHOUT LoRA
```
./outputs/ablation_huav_without_lora/
└── results.json
    ├── metrics: {mae, rmse, mre}
    ├── type_metrics: {depth, distance, ...}
    └── results: [...详细结果...]
```

## 进阶分析

### 查看详细结果

```bash
# WITH LoRA 详细结果
cat ./outputs/ablation_huav_with_lora/results.json | jq .

# WITHOUT LoRA 详细结果
cat ./outputs/ablation_huav_without_lora/results.json | jq .

# 对比特定指标
jq -s '{"with_lora": .[0].metrics, "without_lora": .[1].metrics}' \
    ./outputs/ablation_huav_with_lora/results.json \
    ./outputs/ablation_huav_without_lora/results.json
```

### 可视化 (可选)

创建 Python 脚本绘制对比图表：

```python
import json
import matplotlib.pyplot as plt

# 加载结果
with open('./outputs/ablation_huav_with_lora/results.json') as f:
    with_lora = json.load(f)

with open('./outputs/ablation_huav_without_lora/results.json') as f:
    without_lora = json.load(f)

# 绘制 per-type MRE 对比
types = ['depth', 'distance', 'height', 'length', 'size', 'width']
with_lora_mre = [with_lora['type_metrics'][t]['mre'] for t in types]
without_lora_mre = [without_lora['type_metrics'][t]['mre'] for t in types]

# ... 绘图代码 ...
```

## 注意事项

1. **相同样本**: 两个实验必须使用相同的前 100 个样本
2. **相同设备**: 确保都在 cuda:0 上运行
3. **LoRA 权重**: 确保 LoRA 权重已训练完成
4. **顺序执行**: 避免同时运行（显存限制）

## 故障排除

### LoRA 权重未找到

```bash
❌ LoRA weights not found: ./outputs/lora_injection_sqa/lora_adapters_epochfinal.pt

解决: 先训练 LoRA
./train_lora_injection_sqa.sh
```

### CUDA OOM

```bash
减少 batch size 或使用 8-bit 量化 (已默认启用)
```

### 结果文件缺失

```bash
确保评估脚本成功完成
检查输出目录是否有 results.json
```

## 后续步骤

完成消融实验后，可以：

1. **分析结果**: 确认 LoRA 的有效性
2. **调整超参数**: 如果效果不佳，调整 LoRA rank/layers
3. **扩大实验**: 在全部 17,526 样本上评估
4. **论文撰写**: 使用结果支持论文

## 快速开始

```bash
# 一键运行消融实验
./eval_huav_ablation_with_lora.sh && \
./eval_huav_ablation_without_lora.sh && \
python compare_ablation_results.py
```

---

**Created**: 2025-11-28
**Branch**: `claude/memgen-lora-integration-01Hy2BbsKDVozpNVZkpaGtWU`
**Purpose**: Ablation study for LoRA Memory Weaver effectiveness
