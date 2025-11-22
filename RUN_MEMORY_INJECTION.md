# 运行记忆注入功能 - 完整指南

## 环境要求

### 硬件要求
- **2 张 GPU**（推荐）或 1 张 GPU（测试模式）
- GPU 显存：
  - 使用 `--load_8bit`: 每个 UAV 约 12-16GB
  - 不使用量化: 每个 UAV 约 24-28GB

### 软件要求
```bash
# Python 环境
python >= 3.8

# 关键依赖
torch >= 2.0.0
transformers >= 4.37.0
llava (from source)
```

---

## 快速启动（2 GPU 模式）

### 终端 1: 启动 H-UAV（高资源 UAV）

```bash
# 使用 GPU 0，启动服务器模式
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cuda:0 \
    --port 50051 \
    --model_path ./models/AirSpatialBot \
    --vision_tower /mnt/data/clip-vit-large-patch14-336 \
    --test_data ./data/metadata/airspatial_agent_test_task1.jsonl \
    --image_dir ./data/images \
    --load_8bit \
    --output ./outputs/hierarchical_uav/task1_h-uav_results.jsonl
```

**预期输出**：
```
============================================================
Starting H-UAV Evaluation
============================================================

Loading H-UAV model on cuda:0...
✓ H-UAV model loaded
✓ MAC layers added
✓ Configured as H-UAV (learning mode)

✓ H-UAV Server started
  Listening on port 50051
  Waiting for L-UAV connections...

Press Ctrl+C to stop server
```

---

### 终端 2: 启动 L-UAV（低资源 UAV）

**等待 H-UAV 完全启动后**（看到 "Listening on port 50051"），在新终端运行：

```bash
# 使用 GPU 1，连接到 H-UAV
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device cuda:1 \
    --huav_address localhost:50051 \
    --model_path ./models/AirSpatialBot \
    --vision_tower /mnt/data/clip-vit-large-patch14-336 \
    --test_data ./data/metadata/airspatial_agent_test_task1.jsonl \
    --image_dir ./data/images \
    --threshold 0.7 \
    --load_8bit \
    --output ./outputs/hierarchical_uav/task1_l-uav_results.jsonl
```

**预期输出**：
```
============================================================
Starting L-UAV Evaluation
============================================================

Connecting to H-UAV at localhost:50051...
✓ H-UAV server is reachable

Loading L-UAV model on cuda:1...
✓ L-UAV model loaded
✓ MAC layers added
✓ Configured as L-UAV (inference-only mode)
✓ Self-matching module initialized (threshold=0.7)
✓ Vehicle knowledge base loaded (XXXX records)

Evaluating on XXX samples...
============================================================
L-UAV Evaluation:   0%|          | 0/XXX [00:00<?, ?it/s]

Sample 0: Using H-UAV memory-augmented inference  # ← 记忆注入！
Sample 10: Using local inference without H-UAV memory
Sample 20: Using H-UAV memory-augmented inference  # ← 记忆注入！
...
```

---

## 测试模式（1 GPU 或 CPU）

如果只有 1 张 GPU，可以用测试模式：

### 方案 A: 顺序测试（先 H-UAV 后 L-UAV）

```bash
# 1. 先运行 H-UAV 构建记忆池（处理部分数据）
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cuda:0 \
    --port 50051 \
    --test_data ./data/metadata/airspatial_agent_test_task1.jsonl \
    --image_dir ./data/images \
    --load_8bit

# 按 Ctrl+C 停止后，H-UAV 会保存记忆状态到：
# ./outputs/hierarchical_uav/task1_h-uav_results_memory.pt

# 2. 再运行 L-UAV（可以不连接 H-UAV，只用本地推理）
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device cuda:0 \
    --test_data ./data/metadata/airspatial_agent_test_task1.jsonl \
    --image_dir ./data/images \
    --load_8bit
```

### 方案 B: 同 GPU 模拟（H-UAV 和 L-UAV 都用 cuda:0）

```bash
# 终端 1: H-UAV
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cuda:0 \
    --port 50051 \
    --load_8bit

# 终端 2: L-UAV（同一 GPU，但可能显存不足）
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device cuda:0 \
    --huav_address localhost:50051 \
    --load_8bit
```

---

## 验证记忆注入功能

### 1. 查看 L-UAV 日志

运行时应该看到：

✅ **有记忆注入**（查询 H-UAV）：
```
Sample 10: Using H-UAV memory-augmented inference
Memory injection: added 1 memory tokens with weight 0.5
```

✅ **无记忆**（本地推理）：
```
Sample 20: Using local inference without H-UAV memory
```

### 2. 检查结果统计

L-UAV 运行结束后会显示：

```
============================================================
L-UAV Evaluation Statistics
============================================================
Total samples: 100
Local decisions: 60 (60.0%)
Remote queries: 40 (40.0%)        # ← H-UAV 查询次数

Cache statistics:
  Total cache hits: 15
  Cache hit rate: 37.50%

============================================================
Accuracy Evaluation
============================================================
Overall Accuracy: 85/100 = 85.00%

Accuracy by Source:
  knowledge_base :  50/ 55 = 90.9%
  huav          :  25/ 30 = 83.3%  # ← 使用 H-UAV 记忆的准确率
  local         :  10/ 15 = 66.7%  # ← 纯本地推理的准确率
```

**关键指标**：
- `huav` vs `local` 的准确率应该有差异（验证记忆注入有效）
- Remote queries > 0 说明 L-UAV 在查询 H-UAV

### 3. 检查结果文件

```bash
# L-UAV 结果
cat ./outputs/hierarchical_uav/task1_l-uav_results.jsonl | head -3

# 应该包含 source 字段：
# {"source": "huav", ...}        ← 使用了 H-UAV 记忆
# {"source": "local", ...}       ← 纯本地推理
# {"source": "knowledge_base", ...}  ← 知识库直接查询
```

---

## 调优参数

### Self-Matching 阈值（控制查询频率）

```bash
# 更低阈值 = 更多查询 H-UAV
--threshold 0.5

# 更高阈值 = 更少查询 H-UAV
--threshold 0.9
```

### Memory Weight（控制记忆影响）

在 `eval_task1.py:584` 修改：
```python
memory_weight=0.5   # 默认：平衡图像和记忆
memory_weight=0.3   # 更依赖图像
memory_weight=0.7   # 更依赖记忆
```

### 量化选项

```bash
# 8-bit 量化（推荐，节省显存）
--load_8bit

# 4-bit 量化（实验性，更省显存但可能影响性能）
--load_4bit

# 不使用量化（需要更多显存）
# 不加 --load_8bit 或 --load_4bit
```

---

## 常见问题

### Q1: "Cannot connect to H-UAV"

**检查**：
1. H-UAV 是否已启动？
2. 端口是否正确（默认 50051）？
3. 防火墙是否阻止？

**解决**：
```bash
# 检查 H-UAV 是否在监听
netstat -tlnp | grep 50051

# 或使用 telnet 测试连接
telnet localhost 50051
```

### Q2: CUDA Out of Memory

**解决方案**：
1. 使用 `--load_8bit` 量化
2. 减少 batch size（目前为 1，已最小）
3. 使用单 GPU 顺序模式
4. 减少测试数据量：
   ```bash
   head -n 50 ./data/metadata/airspatial_agent_test_task1.jsonl > test_small.jsonl
   --test_data test_small.jsonl
   ```

### Q3: 没有看到 "memory-augmented inference"

**可能原因**：
1. Self-matching threshold 太高，所有样本都本地处理
2. 知识库覆盖所有样本，直接返回答案

**检查**：
```bash
# 查看 self-matching 统计
# L-UAV 结束时会显示：
Self-matching statistics:
  Query rate: 40.00%        # ← 应该 > 0%
  H-UAV queries: 40
```

### Q4: 想看更详细的日志

```bash
# 添加日志级别参数（需要修改代码）
# 或设置环境变量
export PYTHONUNBUFFERED=1
export LOGLEVEL=DEBUG

# 运行时会显示更多调试信息
```

---

## 与原始 main_task1.py 的对比

### 原始方法（main_task1.py）
```bash
python main_task1.py
```
- 单进程、单 GPU
- 基于 Agent Planning
- 无记忆机制
- 知识库查询基于 3D 尺寸

### 新方法（eval_task1.py with Memory Injection）
```bash
# H-UAV + L-UAV
python hierarchical_uav/eval_task1.py --uav_type h-uav ...
python hierarchical_uav/eval_task1.py --uav_type l-uav ...
```
- 分布式、双 GPU
- 基于 VLM + MAC Memory
- **记忆注入增强推理** ✨
- 知识库查询基于图像 ID + bbox（更精确）

---

## 成功标志

✅ **记忆注入成功运行的标志**：

1. H-UAV 日志显示：
   ```
   [H-UAV Stats] Requests: 40, Cache hit rate: 37.50%
   ```

2. L-UAV 日志显示：
   ```
   Sample 10: Using H-UAV memory-augmented inference
   Memory injection: added 1 memory tokens with weight 0.5
   ```

3. 统计结果显示 `huav` 和 `local` 有性能差异：
   ```
   Accuracy by Source:
     huav   : 25/30 = 83.3%
     local  : 10/15 = 66.7%
   ```

---

## 下一步

完成测试后，可以：

1. **分析结果**：比较有/无记忆注入的性能差异
2. **调优参数**：调整 threshold、memory_weight
3. **实现下一步功能**：H-UAV Test-Time Computing

**祝您测试顺利！🚀**
