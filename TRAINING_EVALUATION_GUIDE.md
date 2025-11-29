# MemGen Weaver 训练与评估指南

## 概述

本指南介绍如何训练 MemGen 风格的记忆编织器，并对比有/无 MemGen 的性能。

---

## 1. 训练 MemGen Weaver

### 1.1 训练脚本功能

训练脚本现在包含以下功能：

✅ **验证集划分**：自动划分 10% 的数据作为验证集
✅ **最佳 Checkpoint 保存**：基于验证损失保存最佳模型
✅ **损失曲线绘制**：自动生成训练和验证损失曲线
✅ **TensorBoard 支持**：实时监控训练过程
✅ **损失历史记录**：导出 JSON 格式的损失历史

### 1.2 运行训练

```bash
# 启动训练
./train_memgen_weaver.sh
```

**训练配置：**
```bash
MODEL_PATH="/mnt/data/AirSpatialBot"
TRAIN_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
OUTPUT_DIR="./outputs/memgen_weaver_sqa"

# MemGen 参数
NUM_MEMORY_TOKENS=8      # Query latents 数量
LORA_RANK=16             # LoRA rank
LORA_ALPHA=32.0          # LoRA alpha
LORA_DROPOUT=0.1         # LoRA dropout

# 训练参数
NUM_EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=1e-5
WEIGHT_DECAY=0.01
VAL_SPLIT=0.1            # 验证集比例 (10%)
```

### 1.3 训练输出

训练完成后，会在 `outputs/memgen_weaver_sqa/` 生成以下文件：

```
outputs/memgen_weaver_sqa/
├── best_checkpoint/              # ✅ 最佳模型（基于验证损失）
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── query_latents.pt
│   └── metadata.json             # 最佳 epoch 信息
├── loss_curves.png               # ✅ 损失曲线图
├── loss_history.json             # ✅ 损失历史数据
├── config.json                   # 训练配置
└── tensorboard/                  # ✅ TensorBoard 日志
    └── events.out.tfevents.*
```

### 1.4 查看训练曲线

#### 方法 1: 查看 PNG 图片

```bash
# 直接查看生成的损失曲线
open outputs/memgen_weaver_sqa/loss_curves.png
```

**曲线包含：**
- 蓝色线：训练损失
- 红色线：验证损失
- 绿色虚线：最佳验证损失的 epoch
- 黄色标注：最佳损失值

#### 方法 2: TensorBoard 实时监控

```bash
# 启动 TensorBoard
tensorboard --logdir=outputs/memgen_weaver_sqa/tensorboard

# 在浏览器中打开
# http://localhost:6006
```

**TensorBoard 视图：**
- `Train/Loss`：每步的训练损失
- `Epoch/Train_Loss`：每个 epoch 的平均训练损失
- `Epoch/Val_Loss`：每个 epoch 的验证损失

#### 方法 3: 查看 JSON 数据

```bash
# 查看损失历史
cat outputs/memgen_weaver_sqa/loss_history.json
```

**JSON 格式：**
```json
{
  "train_losses": [2.3456, 1.8765, 1.5432],
  "val_losses": [2.4567, 2.0123, 1.7890],
  "best_epoch": 3,
  "best_val_loss": 1.7890
}
```

### 1.5 训练过程示例输出

```
======================================================================
Training MemGen-style Memory Weaver
======================================================================

Device: cuda

Creating MemGen Weaver...
MemGen Weaver initialized:
  - Query Latents: 8 tokens
  - LoRA Rank: 16, Alpha: 32.0
  - Target Modules: ['q_proj', 'v_proj']

Trainable parameters:
  ✓ query_latents: 32,768
  ✓ lora_model.base_model.model.layers.8.self_attn.q_proj.lora_A: ...
  ✓ lora_model.base_model.model.layers.8.self_attn.q_proj.lora_B: ...
  ...

Trainable params: 432,768 || All params: 7,000,000,000 || Trainable ratio: 0.01%

Loading data from /mnt/data/AirSpatial/airspatial_sqa_test.jsonl...
Train samples: 900
Validation samples: 100

======================================================================
Starting Training
======================================================================
Epochs: 3
Batch Size: 4
Learning Rate: 1e-5
Validation Split: 0.1
======================================================================

======================================================================
Epoch 1/3
======================================================================
Training: 100%|████████████| 225/225 [12:34<00:00, 3.35s/it, loss=2.1234, avg_loss=2.3456]

📊 Epoch 1 Training Summary:
  Average Training Loss: 2.3456

🔍 Running validation...
Validating: 100%|████████████| 25/25 [01:12<00:00, 2.88s/it]
  Validation Loss: 2.4567

  ✅ New best checkpoint saved! (val_loss: 2.4567)

  📌 Best so far: Epoch 1, Val Loss: 2.4567

======================================================================
Epoch 2/3
======================================================================
...

======================================================================
Training Complete!
======================================================================
Best Validation Loss: 1.7890 (Epoch 3)
Best checkpoint: outputs/memgen_weaver_sqa/best_checkpoint

✓ Loss curve saved to outputs/memgen_weaver_sqa/loss_curves.png
✓ Loss history saved to outputs/memgen_weaver_sqa/loss_history.json

======================================================================
```

---

## 2. 评估对比（有/无 MemGen）

### 2.1 运行评估

```bash
# 评估前 500 个样本
./eval_memgen_comparison.sh
```

### 2.2 评估配置

```bash
MODEL_PATH="/mnt/data/AirSpatialBot"
EVAL_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
MEMGEN_CHECKPOINT="./outputs/memgen_weaver_sqa/best_checkpoint"
OUTPUT_DIR="./outputs/memgen_eval_comparison"
MAX_SAMPLES=500
```

### 2.3 评估指标

评估脚本对比以下指标：

1. **Accuracy（准确率）**
   - Exact match 准确率
   - 衡量完全正确的答案比例

2. **BLEU Score**
   - 基于词重叠的相似度
   - 衡量答案质量

3. **Inference Time（推理时间）**
   - 平均每个样本的推理时间
   - 包括 MemGen 记忆生成时间

### 2.4 评估输出

```
outputs/memgen_eval_comparison/
├── comparison_results.json       # 详细结果
└── memgen_comparison.png         # 对比图表
```

### 2.5 评估示例输出

```
======================================================================
Evaluating H-UAV WITHOUT MemGen (Baseline)
======================================================================
Baseline: 100%|████████████| 500/500 [08:20<00:00, 1.00s/it]

📊 Results (WITHOUT MemGen):
  Samples: 500
  Accuracy: 45.20%
  Average BLEU: 0.5234
  Average Time: 10.5 ms

======================================================================
Evaluating H-UAV WITH MemGen
======================================================================
With MemGen: 100%|████████████| 500/500 [10:15<00:00, 1.23s/it]

📊 Results (WITH MemGen):
  Samples: 500
  Accuracy: 52.80%
  Average BLEU: 0.6012
  Average Time: 22.3 ms

======================================================================
COMPARISON SUMMARY
======================================================================

Metric               Baseline        With MemGen     Improvement
----------------------------------------------------------------------
Accuracy               45.20%          52.80%          +7.60%
BLEU Score            0.5234          0.6012          +0.0778
Avg Time (ms)           10.5            22.3           +11.8
======================================================================

✓ Results saved to outputs/memgen_eval_comparison/comparison_results.json
✓ Comparison chart saved to outputs/memgen_eval_comparison/memgen_comparison.png
```

### 2.6 对比图表

生成的 `memgen_comparison.png` 包含三个子图：

1. **Accuracy Comparison（准确率对比）**
   - 柱状图对比两种方法的准确率
   - 显示提升百分比

2. **BLEU Score Comparison（BLEU 分数对比）**
   - 柱状图对比 BLEU 分数
   - 显示提升数值

3. **Inference Time Comparison（推理时间对比）**
   - 柱状图对比推理时间
   - 显示时间增加量

**图表特点：**
- 绿色/蓝色柱状图
- 数值标注在柱子上方
- 黄色标注显示提升/变化量

---

## 3. 结果分析

### 3.1 预期结果

**有 MemGen 的优势：**
- ✅ 准确率提升：5-10%
- ✅ BLEU 分数提升：0.05-0.10
- ⚠️ 推理时间增加：10-15 ms（但换来更高质量）

**为什么 MemGen 更好？**
1. **可学习 Query Latents**：捕获任务特定的记忆模式
2. **完整 Transformer 处理**：利用自注意力机制
3. **上下文感知**：记忆根据输入动态调整

### 3.2 结果解读

#### Case 1: MemGen 显著提升

```
Accuracy: 45% → 55% (+10%)
BLEU: 0.52 → 0.63 (+0.11)
```

**解释：**
- MemGen 成功学习到任务相关的记忆模式
- Query latents 捕获到关键信息
- 推荐使用 MemGen

#### Case 2: MemGen 轻微提升

```
Accuracy: 45% → 48% (+3%)
BLEU: 0.52 → 0.55 (+0.03)
```

**可能原因：**
- 训练数据不足
- 需要更多 epoch
- 尝试调整 LoRA rank 或 query latents 数量

#### Case 3: MemGen 无明显提升

```
Accuracy: 45% → 46% (+1%)
BLEU: 0.52 → 0.53 (+0.01)
```

**排查步骤：**
1. 检查训练是否收敛（查看 loss curves）
2. 验证 checkpoint 是否正确加载
3. 尝试增加 query latents 数量（8 → 16）
4. 尝试增加 LoRA rank（16 → 32）

---

## 4. 超参数调优

### 4.1 LoRA 参数调优

```bash
# 更强的表达能力
LORA_RANK=32
LORA_ALPHA=64.0

# 更轻量级
LORA_RANK=8
LORA_ALPHA=16.0
```

**建议：**
- 起始值：rank=16, alpha=32（MemGen 标准）
- 如果性能不足：增加到 rank=32
- 如果内存不足：降低到 rank=8

### 4.2 Query Latents 数量调优

```bash
# 更多记忆容量
NUM_MEMORY_TOKENS=16

# 更少参数
NUM_MEMORY_TOKENS=4
```

**建议：**
- 起始值：8 tokens
- 复杂任务：12-16 tokens
- 简单任务：4-8 tokens

### 4.3 训练参数调优

```bash
# 更充分的训练
NUM_EPOCHS=5
BATCH_SIZE=8

# 更快的训练
NUM_EPOCHS=2
BATCH_SIZE=2
```

**建议：**
- 起始值：3 epochs, batch_size=4
- 如果 loss 未收敛：增加 epochs
- 如果内存不足：降低 batch_size

---

## 5. 常见问题

### Q1: 训练时显存不足

**解决方案：**
```bash
# 降低 batch size
BATCH_SIZE=2

# 或使用梯度累积
GRADIENT_ACCUMULATION=2
```

### Q2: 验证损失不下降

**可能原因：**
1. 学习率过大 → 降低到 5e-6
2. 数据过拟合 → 增加更多训练数据
3. 模型配置不当 → 检查 LoRA rank

### Q3: MemGen 评估时内存不足

**解决方案：**
```bash
# 减少评估样本
MAX_SAMPLES=100

# 或使用 CPU（慢）
CUDA_VISIBLE_DEVICES="" python eval_...
```

### Q4: 如何选择最佳 checkpoint？

**答：** 训练脚本会自动选择验证损失最低的 checkpoint，保存在 `best_checkpoint/` 目录。

### Q5: 如何查看 TensorBoard？

```bash
tensorboard --logdir=outputs/memgen_weaver_sqa/tensorboard
# 打开 http://localhost:6006
```

---

## 6. 快速开始流程

### 完整流程

```bash
# 1. 训练 MemGen Weaver
./train_memgen_weaver.sh

# 2. 查看训练结果
cat outputs/memgen_weaver_sqa/loss_history.json
open outputs/memgen_weaver_sqa/loss_curves.png

# 3. 运行评估对比
./eval_memgen_comparison.sh

# 4. 查看对比结果
cat outputs/memgen_eval_comparison/comparison_results.json
open outputs/memgen_eval_comparison/memgen_comparison.png
```

### 预期时间

- **训练**（3 epochs, 1000 样本）：~30-60 分钟
- **评估**（500 样本）：~15-20 分钟
- **总计**：~1-1.5 小时

---

## 7. 输出文件说明

### 训练输出

| 文件 | 说明 |
|------|------|
| `best_checkpoint/` | 最佳模型（基于验证损失） |
| `loss_curves.png` | 损失曲线图（训练 + 验证） |
| `loss_history.json` | 损失历史数据 |
| `config.json` | 训练配置 |
| `tensorboard/` | TensorBoard 日志 |

### 评估输出

| 文件 | 说明 |
|------|------|
| `comparison_results.json` | 详细对比结果（baseline + memgen） |
| `memgen_comparison.png` | 对比图表（3 个子图） |

---

## 8. 总结

### ✅ 训练功能
- 自动验证集划分
- 最佳 checkpoint 保存
- 损失曲线可视化
- TensorBoard 支持

### ✅ 评估功能
- 对比 with/without MemGen
- 多指标评估（Accuracy, BLEU, Time）
- 可视化对比图表
- 详细结果导出

### 🎯 使用建议
1. 先用默认参数训练
2. 查看 loss curves 确认收敛
3. 运行评估对比
4. 根据结果调优超参数

**现在可以开始训练了！** 🚀
