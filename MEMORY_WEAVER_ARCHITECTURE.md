# Memory Weaver Architecture (记忆编织器)

## 核心理解

**记忆编织器 = LoRA 适配器注入冻结的 LLaVA**

```
H-UAV Architecture:
┌──────────────────────────────────────┐
│ Frozen LLaVA (Base Model)            │
│   ├─ Layer 0-7: Frozen               │
│   ├─ Layer 8: q_proj, v_proj ← LoRA │
│   ├─ Layer 9-15: Frozen              │
│   ├─ Layer 16: q_proj, v_proj ← LoRA│
│   ├─ Layer 17-23: Frozen             │
│   ├─ Layer 24: q_proj, v_proj ← LoRA│
│   └─ Layer 25-31: Frozen             │
│                                      │
│ Memory Token Generator               │
│   └─ [seq_len, 4096] → [8, 4096]    │
└──────────────────────────────────────┘
```

---

## 关键设计要点

### 1. 不是独立网络

❌ **错误理解**：记忆编织器是独立的多层网络
✅ **正确理解**：记忆编织器是 LoRA 适配器，挂载在冻结的 LLaVA 上

```python
# ❌ 错误：独立网络
memory_weaver = nn.Sequential(
    nn.Linear(4096, 2048),
    nn.GELU(),
    nn.Linear(2048, 8 * 4096)
)

# ✅ 正确：LoRA 注入
base_llava = load_llava()  # Frozen
for layer_idx in [8, 16, 24]:
    layer = base_llava.layers[layer_idx]
    layer.self_attn.q_proj ← inject_lora(rank=8)
    layer.self_attn.v_proj ← inject_lora(rank=8)
```

### 2. 选择性层注入

❌ **不需要**：每层都注入 LoRA
✅ **正确**：只在特定层注入（如 layers 8, 16, 24）

**原因**：
- 降低参数量
- 均匀分布在模型深度
- 捕获不同抽象层次的特征

**层数选择策略**：
- **浅层 (0-7)**：视觉特征提取，通常不注入
- **中层 (8-16)**：语义理解，注入 LoRA
- **深层 (17-24)**：推理和生成，注入 LoRA

### 3. 目标模块

**常见选择**：
- `q_proj`, `v_proj`：注意力的查询和值投影（推荐）
- `k_proj`：注意力的键投影（可选）
- `o_proj`：注意力输出投影（可选）
- `gate_proj`, `up_proj`, `down_proj`：FFN 层（可选）

**推荐配置**：`q_proj`, `v_proj`（MemGen 使用）

**理由**：
- 注意力机制直接影响信息检索
- Q 和 V 控制"查什么"和"返回什么"
- 参数效率高（仅注入注意力层）

### 4. 深度等同于主 LLM

❌ **错误**：记忆编织器有自己的层数
✅ **正确**：深度 = LLaVA 层数，但只在特定层微调

```
LLaVA: 32 layers total
Memory Weaver: Same 32 layers
  - Layers 0-7, 9-15, 17-23, 25-31: Frozen
  - Layers 8, 16, 24: LoRA injected (trainable)

Trainable parameters: ~1-2% of total
```

---

## 工作流程

### H-UAV 端（记忆生成）

```
1. Receive L-UAV hidden states
   Input: [batch_size, seq_len, 4096]
   ↓
2. Process through LoRA-injected LLaVA
   ├─ Frozen layers: Use original weights
   ├─ LoRA layers: Apply low-rank adaptation
   │   y = W_0 @ x + (B @ A) @ x * (alpha / rank)
   └─ Output: Enhanced hidden states
   ↓
3. Memory Token Generator
   ├─ Pool: [batch, seq_len, 4096] → [batch, 4096]
   ├─ Project: [batch, 4096] → [batch, 8 * 4096]
   └─ Reshape: [batch, 8 * 4096] → [batch, 8, 4096]
   ↓
4. Return memory tokens
   Output: [batch_size, 8, 4096]
```

### L-UAV 端（记忆使用）

```
1. Base inference → low confidence
   ↓
2. Send hidden states to H-UAV
   POST /get_memory
   Body: {hidden_states: [...], shape: [1, 64, 4096]}
   ↓
3. Receive memory tokens from H-UAV
   memory_tokens: [8, 4096]
   ↓
4. Concatenate with input
   enhanced_hidden = concat([memory_tokens, input_tokens])
   Shape: [8 + 64, 4096] = [72, 4096]
   ↓
5. Enhanced inference
   LLaVA.generate(enhanced_hidden) → answer
```

---

## LoRA 参数

### 典型配置

```python
config = LoRAInjectionConfig(
    lora_rank=8,              # Low-rank dimension
    lora_alpha=16.0,          # Scaling factor (usually 2 * rank)
    lora_dropout=0.1,         # Dropout for regularization
    target_modules=['q_proj', 'v_proj'],
    target_layers=[8, 16, 24],
    num_memory_tokens=8
)
```

### 参数量计算

**单个 LoRA 适配器**：
```
in_features = 4096
out_features = 4096
rank = 8

LoRA_A: 4096 × 8 = 32,768 params
LoRA_B: 8 × 4096 = 32,768 params
Total: 65,536 params per adapter

vs Original Linear: 4096 × 4096 = 16,777,216 params
Reduction: 65,536 / 16,777,216 = 0.39% (256× smaller!)
```

**完整记忆编织器**：
```
Target layers: 3 (layers 8, 16, 24)
Target modules per layer: 2 (q_proj, v_proj)
Total adapters: 3 × 2 = 6

LoRA parameters: 6 × 65,536 = 393,216 (~0.4 MB)
Memory Token Generator: ~8M params (~32 MB)
Total: ~8.4M trainable params (~34 MB)

vs Full LLaVA: 7B params (~28 GB)
Trainable ratio: 0.12%
```

---

## 为什么这样设计？

### 1. 参数效率

| Component | Params | Size | Trainable |
|-----------|--------|------|-----------|
| Frozen LLaVA | 7B | 28 GB | ❌ |
| LoRA adapters (6) | 393K | 1.5 MB | ✅ |
| Memory Generator | 8M | 32 MB | ✅ |
| **Total trainable** | **8.4M** | **34 MB** | **0.12%** |

**好处**：
- 训练快（只更新 0.12% 参数）
- 部署轻（只需传输 34 MB LoRA 权重）
- 可切换（多个任务对应不同 LoRA）

### 2. 动态性

基于 **L-UAV 已生成 token 的 hidden states** 生成记忆：

```
L-UAV context: "The depth of the building is..."
                   ↓ (hidden states)
H-UAV LoRA: Process context-aware features
                   ↓
Memory tokens: Contextual, dynamic memory
```

vs 静态记忆池：
```
Static memory: Pre-computed embeddings
               ↓
Not context-aware, fixed representations
```

### 3. MemGen 对齐

参考 **MemGen** 设计：
- LoRA 适配器（不是独立网络）
- 选择性层注入
- 潜变量记忆生成
- 轻量级、可训练

---

## 训练策略

### 训练目标

**目标**：学习生成有用的记忆 tokens，使 L-UAV 的增强推理更准确。

**损失函数**：
```python
# Option 1: 端到端训练（需要 L-UAV 参与）
loss = CrossEntropyLoss(
    luav_enhanced_answer,
    ground_truth
)

# Option 2: 对比学习
loss = ContrastiveLoss(
    positive_memory_tokens,
    negative_memory_tokens
)

# Option 3: 重建损失
loss = MSE(
    reconstructed_hidden,
    original_hidden
)
```

### 训练流程

```bash
# 1. 准备数据
# 收集 (L-UAV hidden states, ground truth) pairs

# 2. 训练 LoRA adapters
python hierarchical_uav/train_lora_injection.py \
    --model_path /mnt/data/AirSpatialBot \
    --train_data /mnt/data/AirSpatial/airspatial_sqa_test.jsonl \
    --output_dir ./outputs/lora_injection_sqa \
    --lora_rank 8 \
    --target_layers 8 16 24 \
    --num_epochs 3

# 3. 保存 LoRA 权重
# outputs/lora_injection_sqa/lora_adapters_final.pt (~34 MB)

# 4. 部署到 H-UAV
python hierarchical_uav/huav_lora_server_v2.py \
    --lora_weights ./outputs/lora_injection_sqa/lora_adapters_final.pt \
    --target_layers 8 16 24
```

---

## API 接口

### H-UAV Server

```python
POST /get_memory

# Request (Option 1: Hidden states)
{
    "hidden_states": [...],  # L-UAV hidden states
    "shape": [1, 64, 4096]
}

# Request (Option 2: Image + Question)
{
    "image": "base64_encoded_image",
    "question": "What is the depth?"
}

# Response
{
    "memory_tokens": [[...], [...], ...],  # [8, 4096]
    "memory_shape": [8, 4096],
    "memory_generated": true,
    "inference_time_ms": 12.3
}
```

### L-UAV Client

```python
import requests
import torch

# L-UAV generates hidden states
hidden_states = luav_encode(image, question)  # [1, 64, 4096]

# Request H-UAV memory
response = requests.post('http://h-uav:8000/get_memory', json={
    'hidden_states': hidden_states.tolist(),
    'shape': list(hidden_states.shape)
})

# Receive memory tokens
memory_tokens = torch.tensor(response.json()['memory_tokens'])  # [8, 4096]

# Enhanced inference
enhanced_hidden = torch.cat([memory_tokens, hidden_states[0]], dim=0)  # [72, 4096]
answer = luav_generate(enhanced_hidden)
```

---

## 对比总结

| Aspect | 独立网络 | LoRA 注入 (正确) |
|--------|----------|------------------|
| **架构** | 独立多层 MLP | 挂载在 LLaVA 上 |
| **基础模型** | 不需要 | 冻结 LLaVA |
| **层数** | 自定义 (2-4 层) | LLaVA 层数 (32) |
| **参数量** | ~50M | ~8.4M |
| **训练** | 从头训练 | 只训练 LoRA |
| **深度** | 浅 (4 层) | 深 (32 层) |
| **上下文** | 有限 | 完整 LLaVA 理解 |

---

## 最佳实践

### 1. 层选择

**推荐**：均匀分布 + 偏重深层
```python
# 32层 LLaVA，选择 3 层
target_layers = [8, 16, 24]  # 均匀分布

# 或偏重深层（推理能力）
target_layers = [16, 20, 24]
```

### 2. Rank 选择

| Rank | Params | Quality | Use Case |
|------|--------|---------|----------|
| 4 | 196K | 低 | 极限轻量 |
| 8 | 393K | 中 | 推荐（平衡） |
| 16 | 786K | 高 | 性能优先 |
| 32 | 1.5M | 很高 | 复杂任务 |

**推荐**：rank=8 (MemGen 使用)

### 3. 模块选择

**最小配置**：`q_proj, v_proj` (2 modules)
**标准配置**：`q_proj, k_proj, v_proj` (3 modules)
**完整配置**：`q_proj, k_proj, v_proj, o_proj` (4 modules)

**推荐**：`q_proj, v_proj` (MemGen 风格)

---

## 总结

**核心理解**：
1. 记忆编织器 **不是** 独立网络
2. 记忆编织器 **是** LoRA 适配器注入冻结 LLaVA
3. 只在特定层（如 8, 16, 24）注入
4. 深度等于 LLaVA，但只微调 0.12% 参数
5. 基于 L-UAV hidden states 动态生成记忆
6. 返回潜变量记忆序列（latent tokens）

**关键优势**：
- ✅ 参数高效（0.12% trainable）
- ✅ 上下文感知（完整 LLaVA 理解力）
- ✅ 动态生成（基于 L-UAV 状态）
- ✅ 轻量部署（34 MB LoRA 权重）
- ✅ 对齐 MemGen 设计

这才是真正的 **Memory Weaver**！🎯
