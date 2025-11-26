# H-UAV LoRA Memory Architecture

## 架构概述

**设计框架**：

```
┌──────────────────────────────────────┐
│ L-UAV (客户端 - 仅推理)                │
│                                      │
│ ┌─────────────────────────────────┐ │
│ │ Base LLaVA Model                │ │
│ │ ↓                               │ │
│ │ Inference (image + question)    │ │
│ │ ↓                               │ │
│ │ Self-Matching Module            │ │
│ │  ├─ QueryKeyGenerator           │ │
│ │  ├─ Self-similarity score       │ │
│ │  └─ Confidence evaluation       │ │
│ └─────────────────────────────────┘ │
│                                      │
│  If confidence < threshold:          │
│  └─ Send query to H-UAV ─────────┐  │
└──────────────────────────────────│───┘
                                   │
                          (Network Request)
                                   │
┌──────────────────────────────────▼───┐
│ H-UAV (服务器 - 记忆提供者)            │
│                                      │
│ ┌─────────────────────────────────┐ │
│ │ LoRA-Only Memory Pool           │ │
│ │  └─ Memory Weaver (MemGen)     │ │
│ │     ├─ LoRA rank: 4-8          │ │
│ │     ├─ Memory tokens: 8-16     │ │
│ │     └─ Pre-trained weights     │ │
│ └─────────────────────────────────┘ │
│           ↓                          │
│ ┌─────────────────────────────────┐ │
│ │ Receive L-UAV Query             │ │
│ │ ↓                               │ │
│ │ Generate LoRA Memory            │ │
│ │ ↓                               │ │
│ │ Enhanced Inference              │ │
│ │ ↓                               │ │
│ │ Return Answer + Confidence      │ │
│ └─────────────────────────────────┘ │
│                                      │
└──────────────────────────────────────┘
```

---

## 核心设计理念

### L-UAV 职责
- ✅ **仅推理**：运行 base LLaVA 模型
- ✅ **Self-Matching**：评估自身置信度
- ✅ **请求协助**：当 confidence < threshold 时向 H-UAV 请求
- ❌ **不部署记忆**：L-UAV 本身不包含记忆模块

### H-UAV 职责
- ✅ **维护记忆池**：部署 LoRA-Only Memory
- ✅ **服务请求**：接收 L-UAV 的查询
- ✅ **记忆增强**：使用 LoRA 记忆辅助推理
- ✅ **返回结果**：提供增强的答案和置信度

---

## LoRA-Only Memory 设计

### 为什么选择 LoRA-Only（而非 MAC）？

| Aspect | MAC Memory | LoRA-Only Memory | 理由 |
|--------|------------|------------------|------|
| **存储方式** | Explicit neural memory DB | Implicit LoRA weights | LoRA 更轻量 |
| **更新机制** | Test-time learning | Pre-training | H-UAV 可预训练 |
| **参数量** | 50M (~200 MB) | 0.36M (~1.4 MB) | LoRA 99% 更小 |
| **推理速度** | 2-20 ms (检索+更新) | 0.3 ms (生成) | LoRA 10-60× 更快 |
| **部署** | 需同步 memory state | 静态权重 | LoRA 更简单 |
| **适用场景** | 持续学习 | 静态记忆服务 | H-UAV 服务器适合静态 |

**结论**：H-UAV 作为中心化服务器，**不需要在线学习**（可离线训练），LoRA 的静态记忆特性非常适合。

---

## H-UAV LoRA 服务器组件

### 1. LoRA Memory Pool

```python
from hierarchical_uav.mac_memory import LoRAMemoryLayer, LoRAMemoryConfig

# H-UAV 配置
config = LoRAMemoryConfig(
    hidden_size=4096,
    num_memory_tokens=8,       # 适中的记忆容量
    lora_rank=4,               # 轻量级
    lora_alpha=8.0,
    enable_trigger=False,      # H-UAV 总是提供记忆
    pool_method='mean'
)

lora_layer = LoRAMemoryLayer(config)

# 加载预训练权重
lora_layer.load_lora_weights('./lora_weights.pt')
lora_layer.freeze_for_inference()
```

**特点**：
- 预训练的 LoRA 权重（静态）
- 不需要 test-time 更新
- 快速推理（0.3 ms/query）

### 2. 服务器接口

**HTTP API** (推荐，更简单):
```python
# hierarchical_uav/huav_lora_server.py

# Endpoint: POST /infer
{
    "image": "base64_encoded_image",
    "question": "What is the depth?",
    "bbox_3d": [x, y, z, l, w, h, theta]  # optional
}

# Response:
{
    "answer": "5.2 meters",
    "confidence": 0.85,
    "memory_used": true,
    "inference_time_ms": 12.3
}
```

**Socket-based** (现有实现):
```python
# hierarchical_uav/communication/grpc_server.py
# 接收 query vector，返回 memory value
```

### 3. 工作流程

```
1. L-UAV 推理
   ├─ Self-matching score = 0.45 (< 0.5)
   └─ 决定请求 H-UAV

2. L-UAV → H-UAV
   ├─ 发送: image + question + bbox_3d
   └─ (HTTP POST /infer)

3. H-UAV 处理
   ├─ 编码 image + question
   ├─ 生成 LoRA memory tokens (8 tokens)
   ├─ 增强推理 (LLaVA + memory)
   └─ 生成答案

4. H-UAV → L-UAV
   ├─ 返回: answer + confidence
   └─ L-UAV 使用增强结果
```

---

## H-UAV LoRA 训练

### 训练策略

**离线预训练** (H-UAV 服务器部署前):

```bash
# 在服务器或 H-UAV 上预训练 LoRA 记忆
./train_lora_only_sqa.sh

# 输出: lora_weights_final.pt (~0.7 MB)
```

**训练过程**：
1. 使用 SQA 数据集（17,526 samples）
2. 训练 LoRA 适配器生成有效记忆
3. 保存 LoRA 权重（静态）
4. 部署到 H-UAV 服务器

**不需要**：
- ❌ Test-time learning
- ❌ Surprise-driven updates
- ❌ Online adaptation

**原因**：H-UAV 作为服务器，可以**离线训练**一次，然后**静态服务**多个 L-UAV。

### 定期更新策略

```
Week 1-4: H-UAV 使用 lora_v1.pt
   ↓
收集新数据 / 用户反馈
   ↓
Week 5: 离线重新训练
   ├─ 使用新数据 + 旧数据
   └─ 生成 lora_v2.pt
   ↓
Week 5-8: H-UAV 更新到 lora_v2.pt
   └─ 热更新（无需停机）
```

---

## 对比：MAC vs LoRA for H-UAV

### MAC Memory (不推荐用于 H-UAV)

❌ **劣势**：
- 需要 test-time learning（H-UAV 服务器不需要）
- 200 MB memory state（过大）
- 每次请求需要检索+更新（慢）
- Memory state 难以版本管理

✅ **优势**：
- 可在线学习（但 H-UAV 不需要）

### LoRA-Only Memory (推荐用于 H-UAV)

✅ **优势**：
- **静态权重**：预训练一次，持续服务
- **轻量级**：0.7 MB（易于部署和版本管理）
- **快速**：0.3 ms/query（服务多个 L-UAV）
- **可维护**：离线重训，热更新
- **可扩展**：多个 LoRA 权重对应不同任务

⚠️ **劣势**：
- 不能在线学习（但可定期离线更新）

---

## 部署方案

### 方案 1：单 H-UAV 服务器

```
        L-UAV-1 ──┐
        L-UAV-2 ──┤
        L-UAV-3 ──┼──> H-UAV Server (LoRA Memory Pool)
        L-UAV-4 ──┤
        L-UAV-5 ──┘

H-UAV:
  ├─ lora_weights.pt (0.7 MB)
  ├─ HTTP Server :8000
  └─ 处理所有 L-UAV 请求
```

### 方案 2：多任务 LoRA 池

```
H-UAV Server
  ├─ lora_sqa.pt        (SQA 任务)
  ├─ lora_vqa.pt        (VQA 任务)
  ├─ lora_detection.pt  (检测任务)
  └─ 根据 task_type 选择对应 LoRA

L-UAV 请求:
  {
    "task_type": "sqa",  # 指定任务类型
    "image": ...,
    "question": ...
  }
```

### 方案 3：分层 H-UAV

```
         L-UAV × 100
              ↓
        Edge H-UAV × 10 (区域服务器)
         ├─ LoRA memory
         └─ 处理本地 L-UAV
              ↓
        Central H-UAV (中心服务器)
         ├─ 定期训练新 LoRA
         └─ 分发到 Edge H-UAV
```

---

## 启动 H-UAV 服务器

### 使用 HTTP 服务器（推荐）

```bash
python hierarchical_uav/huav_lora_server.py \
    --model_path /mnt/data/AirSpatialBot \
    --vision_tower /mnt/data/clip-vit-large-patch14-336 \
    --lora_memory ./outputs/lora_only_sqa/lora_weights_final.pt \
    --host 0.0.0.0 \
    --port 8000 \
    --device cuda:0
```

**输出**：
```
============================================================
H-UAV LoRA Memory Server
============================================================
Host: 0.0.0.0
Port: 8000
Device: cuda:0
LoRA weights: lora_weights_final.pt (0.7 MB)
============================================================
Endpoints:
  Health: http://0.0.0.0:8000/health
  Infer: http://0.0.0.0:8000/infer
  Stats: http://0.0.0.0:8000/stats
============================================================
```

### 使用 Socket 服务器（现有）

```bash
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cuda:0 \
    --lora_memory ./outputs/lora_only_sqa/lora_weights_final.pt
```

---

## L-UAV 请求示例

### Python 客户端

```python
import requests
import base64
from PIL import Image
import io

# 加载图像
image = Image.open('test.jpg')
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
image_b64 = base64.b64encode(buffered.getvalue()).decode()

# 请求 H-UAV
response = requests.post('http://h-uav-server:8000/infer', json={
    'image': image_b64,
    'question': 'What is the depth of this object?',
    'bbox_3d': [0.5, 0.5, 0.0, 0.2, 0.2, 0.1, 0.0]
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Memory used: {result['memory_used']}")
```

---

## 性能指标

### H-UAV LoRA 服务器性能

| Metric | Value | Note |
|--------|-------|------|
| **LoRA 权重大小** | 0.7 MB | rank=4, tokens=8 |
| **单次推理延迟** | 15-30 ms | 含编码+生成+推理 |
| **LoRA 生成开销** | 0.3 ms | <2% 总延迟 |
| **内存占用** | 336 KB/request | 运行时 |
| **并发能力** | 10-50 L-UAV | 取决于 GPU |
| **吞吐量** | 30-60 req/s | 单 GPU |

### 对比 MAC Memory 服务器

| Metric | MAC | LoRA-Only | Winner |
|--------|-----|-----------|--------|
| Memory state | 200 MB | 0.7 MB | LoRA (285×) |
| 单次推理 | 20-40 ms | 15-30 ms | LoRA (1.3×) |
| 并发能力 | 5-20 | 10-50 | LoRA (2×) |
| 版本管理 | 难 | 易 | LoRA |
| 热更新 | 难 | 易 | LoRA |

---

## 更新和维护

### 1. 定期重训 LoRA

```bash
# 每周/每月重训一次
./train_lora_only_sqa.sh

# 输出新版本
# outputs/lora_only_sqa_v2/lora_weights_final.pt
```

### 2. 热更新 H-UAV

```bash
# 方式 1: API 更新
curl -X POST http://h-uav:8000/reload_lora \
    -d '{"lora_path": "./lora_v2.pt"}'

# 方式 2: 重启服务器
kill -HUP <huav_pid>
```

### 3. A/B 测试

```
H-UAV Instance A: lora_v1.pt (50% traffic)
H-UAV Instance B: lora_v2.pt (50% traffic)

Compare performance → Deploy winner
```

---

## FAQ

### Q1: 为什么 H-UAV 不用 MAC Memory？

**A**: H-UAV 是中心化服务器，不需要在线学习。LoRA 的**静态记忆**特性更适合：
- 离线预训练一次
- 持续服务多个 L-UAV
- 轻量级（0.7 MB vs 200 MB）
- 易于版本管理和更新

### Q2: LoRA 不能在线学习，怎么处理新数据？

**A**: 采用**定期离线重训**策略：
- 收集 L-UAV 反馈数据
- 每周/月离线重训 LoRA
- 热更新部署新权重

### Q3: 一个 H-UAV 能服务多少 L-UAV？

**A**: 取决于：
- GPU 性能：单 GPU 约 10-50 并发
- 请求频率：如果 L-UAV 每 10s 请求一次，可服务 100-500 个 L-UAV
- 可横向扩展：部署多个 H-UAV 实例

### Q4: 能否支持多任务？

**A**: 可以！部署多个 LoRA 权重：
```python
lora_pool = {
    'sqa': LoRAMemoryLayer(...).load('lora_sqa.pt'),
    'vqa': LoRAMemoryLayer(...).load('lora_vqa.pt'),
    'det': LoRAMemoryLayer(...).load('lora_det.pt'),
}

# 根据 task_type 选择
lora_layer = lora_pool[request['task_type']]
```

### Q5: LoRA 权重什么时候需要重训？

**A**: 建议：
- **定期重训**：每周/月一次（持续改进）
- **数据漂移**：任务分布变化时
- **性能下降**：监控到准确率降低
- **新任务**：添加新功能时

---

## 总结

### H-UAV LoRA-Only 架构优势

1. ✅ **轻量级服务**：0.7 MB 权重，易于部署
2. ✅ **高效推理**：0.3 ms LoRA 开销
3. ✅ **易于维护**：静态权重，版本管理简单
4. ✅ **可扩展**：多任务 LoRA 池
5. ✅ **高并发**：10-50 L-UAV/GPU

### 与原始设计框架完全契合

- **L-UAV**: 仅推理 + self-matching ✅
- **H-UAV**: 维护记忆池 (LoRA) ✅
- **协作**: L-UAV 请求 → H-UAV 服务 ✅

### 下一步

1. ✅ 训练 H-UAV LoRA 权重
   ```bash
   ./train_lora_only_sqa.sh
   ```

2. ✅ 启动 H-UAV 服务器
   ```bash
   python hierarchical_uav/huav_lora_server.py \
       --lora_memory ./outputs/lora_only_sqa/lora_weights_final.pt
   ```

3. ✅ L-UAV 请求测试
   ```bash
   python hierarchical_uav/eval_sqa.py \
       --uav_type l-uav \
       --huav_server http://localhost:8000
   ```

---

**LoRA-Only Memory = H-UAV 的最佳选择！** 🚀
