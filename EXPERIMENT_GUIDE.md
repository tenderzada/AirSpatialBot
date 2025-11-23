# L-UAV 记忆增强实验指南

## 实验目标

测试 H-UAV 记忆系统对 L-UAV 性能的影响，对比：
- **Baseline**: L-UAV 单独推理（无记忆增强）
- **Experiment A**: L-UAV + 未训练的 MAC 记忆
- **Experiment B**: L-UAV + 训练后的 MAC 记忆

## 配置参数说明

### THRESHOLD（Self-Matching 阈值）
```bash
THRESHOLD = 0.0   # 禁用 H-UAV 查询（始终本地推理）
THRESHOLD = 0.7   # 标准阈值（低于 0.7 时查询 H-UAV）
THRESHOLD = 1.0   # 强制查询 H-UAV（几乎所有样本）
```

**工作原理**：
- Self-matching 计算相似度 score ∈ [0, 1]
- 判断条件：`should_query = (score < threshold)`
- 低分 = 不确定 → 查询 H-UAV
- 高分 = 自信 → 本地处理

### FORCE_QUERY_RATE（强制查询比率）
```bash
FORCE_QUERY_RATE = 0.0   # 正常模式（优先使用知识库）
FORCE_QUERY_RATE = 0.3   # 30% 样本跳过知识库，测试记忆
FORCE_QUERY_RATE = 1.0   # 100% 样本跳过知识库，纯 VLM 推理
```

**工作原理**：
- 控制跳过知识库（KB）的样本比例
- `force_query_huav=True` 的样本直接进入 VLM 推理
- 这些样本会经过 self-matching 决定是否查询 H-UAV

### 配置组合效果

| THRESHOLD | FORCE_QUERY_RATE | 预期行为 |
|-----------|------------------|---------|
| 0.0 | 0.0 | Baseline：纯本地推理，KB 优先 |
| 1.0 | 0.3 | 30% + α 样本使用记忆（α 取决于 KB 覆盖率） |
| 1.0 | 1.0 | 100% 样本使用记忆（跳过 KB，强制查询 H-UAV） |

## 实验流程

### Baseline: L-UAV 单独推理

**目的**：建立性能基线，不使用 H-UAV 记忆

```bash
# 运行 L-UAV standalone 模式
./run_luav_standalone.sh
```

**配置**：
- `THRESHOLD=0.0` → 禁用 H-UAV 查询
- `FORCE_QUERY_RATE=0.0` → 使用知识库
- 不需要启动 H-UAV 服务器

**输出**：
- `outputs/hierarchical_uav/task1_l-uav_standalone.jsonl`
- 每个样本的 `source` 应该是 `local` 或 `knowledge_base`

**分析**：
```bash
python analyze_uav_results.py outputs/hierarchical_uav/task1_l-uav_standalone.jsonl
```

---

### Experiment A: L-UAV + 未训练的 MAC 记忆

**目的**：测试随机初始化的记忆系统

**步骤 1：启动 H-UAV（随机初始化）**
```bash
# 在 GPU 0 上启动 H-UAV，不加载训练好的记忆
./run_huav.sh
```

**步骤 2：运行 L-UAV**
```bash
# 在 GPU 1 上运行 L-UAV
./run_luav.sh
```

**配置**：
- `THRESHOLD=1.0` → 强制查询 H-UAV
- `FORCE_QUERY_RATE=0.3` → 30% 样本跳过 KB

**输出**：
```bash
# 保存结果
mv outputs/hierarchical_uav/task1_l-uav_results.jsonl \
   outputs/hierarchical_uav/task1_l-uav_UNTRAINED.jsonl
```

---

### Experiment B: L-UAV + 训练后的 MAC 记忆

**目的**：测试训练后的记忆系统是否提升性能

**步骤 1：启动 H-UAV（加载训练好的记忆）**
```bash
# 重启 H-UAV，加载训练好的 checkpoint
# 先停止之前的 H-UAV
pkill -f "hierarchical_uav/eval_task1.py.*h-uav"

# 启动带有训练记忆的 H-UAV
LOAD_MEMORY=./outputs/huav_training/huav_memory_final.pt ./run_huav.sh
```

**步骤 2：运行 L-UAV**
```bash
# 配置相同，确保对比公平
./run_luav.sh
```

**输出**：
```bash
# 保存结果
mv outputs/hierarchical_uav/task1_l-uav_results.jsonl \
   outputs/hierarchical_uav/task1_l-uav_TRAINED.jsonl
```

---

## 结果分析

### 单独分析

```bash
# 分析 Baseline
python analyze_uav_results.py outputs/hierarchical_uav/task1_l-uav_standalone.jsonl

# 分析 Experiment A
python analyze_uav_results.py outputs/hierarchical_uav/task1_l-uav_UNTRAINED.jsonl

# 分析 Experiment B
python analyze_uav_results.py outputs/hierarchical_uav/task1_l-uav_TRAINED.jsonl
```

### 对比分析

**Baseline vs Memory-Augmented:**
```bash
python compare_baseline_vs_memory.py \
    outputs/hierarchical_uav/task1_l-uav_standalone.jsonl \
    outputs/hierarchical_uav/task1_l-uav_TRAINED.jsonl
```

**Untrained vs Trained Memory:**
```bash
python compare_memory_experiments.py \
    outputs/hierarchical_uav/task1_l-uav_UNTRAINED.jsonl \
    outputs/hierarchical_uav/task1_l-uav_TRAINED.jsonl
```

---

## 预期结果指标

### 记忆使用率

**Baseline:**
- `memory_used`: 0（不使用记忆）
- `source`: 主要是 `knowledge_base` 和 `local`

**Experiment A & B (FORCE_QUERY_RATE=0.3):**
- `memory_used`: 约 30-65%（取决于 KB 覆盖率）
- `source`: 包含 `huav`、`knowledge_base`、`local`

### 准确率

假设 Baseline 准确率为 **A_base**：

**理想情况**：
- Experiment A（未训练记忆）：准确率 ≈ A_base 或略低
  - 原因：随机记忆可能引入噪声
- Experiment B（训练记忆）：准确率 > A_base
  - 原因：训练记忆包含有用的知识

**验证记忆有效性**：
```
Improvement = A_trained - A_base > 0
```

如果 `Improvement > 0`，说明训练的 MAC 记忆确实有助于提升 L-UAV 性能。

---

## 故障排查

### 问题 1：Memory Usage = 0

**症状**：结果中所有样本的 `source` 都不是 `huav`

**可能原因**：
1. H-UAV 服务器未运行
2. gRPC 连接失败
3. 知识库覆盖率 100%（FORCE_QUERY_RATE 太低）

**解决方案**：
```bash
# 检查 H-UAV 是否运行
nc -zv localhost 50051

# 增加 FORCE_QUERY_RATE 到 1.0
# 修改 run_luav.sh: FORCE_QUERY_RATE=1.0

# 查看 L-UAV 日志
grep -E "(H-UAV|should_query|memory)" outputs/hierarchical_uav/luav.log
```

### 问题 2：准确率异常低（< 5%）

**症状**：所有实验准确率都很低

**可能原因**：
1. Ground truth 格式不匹配
2. 答案提取逻辑错误
3. 评估指标不正确

**解决方案**：
```bash
# 检查样本输出
jq '.[0]' outputs/hierarchical_uav/task1_l-uav_results.jsonl

# 检查 answer 和 ground_truth 格式
jq '.answer, .ground_truth' outputs/hierarchical_uav/task1_l-uav_results.jsonl | head -20
```

### 问题 3：H-UAV 和 L-UAV 都在 GPU 0

**症状**：OOM 或性能问题

**解决方案**：
```bash
# 确认设备分配
# run_huav.sh: --device cuda:0
# run_luav.sh: --device cuda:1

# 检查进程
nvidia-smi
```

---

## 成本估算

假设测试集有 **N=1000** 个样本：

| 实验 | KB 查询 | VLM 推理 | H-UAV 查询 | 估计时间（RTX 4090） |
|------|---------|----------|-----------|---------------------|
| Baseline | ~800 | ~200 | 0 | ~10 分钟 |
| Exp A/B (FORCE=0.3) | ~560 | ~440 | ~440 | ~20 分钟 |
| Exp A/B (FORCE=1.0) | 0 | 1000 | 1000 | ~45 分钟 |

**建议**：
- 首先运行 Baseline（快速，低成本）
- 然后运行 FORCE=0.3 的实验（中等成本）
- 如果记忆使用率为 0，再尝试 FORCE=1.0（高成本，但确保测试记忆）

---

## 检查清单

实验前检查：
- [ ] H-UAV 服务器正在运行（`nc -zv localhost 50051`）
- [ ] GPU 内存充足（`nvidia-smi`）
- [ ] 输出目录存在（`mkdir -p outputs/hierarchical_uav`）
- [ ] 配置正确（检查 `THRESHOLD` 和 `FORCE_QUERY_RATE`）

实验后检查：
- [ ] 结果文件已生成
- [ ] 记忆使用率 > 0（非 Baseline）
- [ ] 样本数量正确
- [ ] 准确率在合理范围（> 5%）

---

## 快速测试（调试用）

如果想快速验证配置是否正确：

```bash
# 修改测试数据数量（仅测试 10 个样本）
head -10 /mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl > /tmp/test_small.jsonl

# 运行快速测试
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device cuda:1 \
    --huav_address localhost:50051 \
    --threshold 1.0 \
    --force-query-rate 1.0 \
    --model_path /mnt/data/AirSpatialBot \
    --vision_tower /mnt/data/clip-vit-large-patch14-336 \
    --test_data /tmp/test_small.jsonl \
    --image_dir /mnt/data/AirSpatial/images \
    --load_8bit \
    --output /tmp/test_results.jsonl

# 检查结果
jq '.source' /tmp/test_results.jsonl | sort | uniq -c
```

如果看到 `huav` 出现，说明记忆系统工作正常！
