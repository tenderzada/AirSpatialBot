# MAC Memory Optimization Summary

## 🎯 问题

在 RTX 4090 (24GB) 上启动 H-UAV 时，MAC 记忆层占用了 **9.1 GB**，导致显存不足。

**根本原因：** `memory_value_proj` 层的参数量过大
```
memory_value_proj 参数 = memory_dim × hidden_size × num_memory_tokens
                        = 4096 × 4096 × 128
                        = 2,147,483,648 参数
                        = 8.00 GB (float32)
```

---

## ✅ 优化方案

### 修改的文件：

1. **`hierarchical_uav/models/uav_config.py`**
   - `num_memory_tokens: 128 → 32`

2. **`hierarchical_uav/mac_memory/mac_layer.py`**
   - `num_memory_tokens: 128 → 32`

3. **`hierarchical_uav/mac_memory/neural_memory.py`**
   - `cache_size: 1000 → 500`

### 优化效果：

| 项目 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| **num_memory_tokens** | 128 | 32 | -75% |
| **cache_size** | 1000 | 500 | -50% |
| **总内存占用** | **8.47 GB** | **2.45 GB** | **-71%** ⭐ |
| **memory_value_proj** | 8.00 GB | 2.00 GB | -75% |

---

## 📊 内存分配分析

### 优化后的 MAC 层内存分配 (总计 2.45 GB)：

```
Component                    Parameters      Size
─────────────────────────────────────────────────
memory_value_proj (主要)     537M params    2.00 GB  (82%)
Q/K/V/O projections           67M params    256 MB   (10%)
NeuralMemory MLP              34M params    128 MB   (5%)
memory_query_proj             17M params     64 MB   (2.5%)
Episodic Cache (runtime)      -              16 MB   (0.6%)
Other (persistent, LN)        <1M params      1 MB   (0.04%)
─────────────────────────────────────────────────
Total                         655M params    2.45 GB
```

---

## 🖥️ RTX 4090 (24GB) 内存预算

```
Component                     Memory
──────────────────────────────────────
LLaVA-7B (8-bit)              ~8.0 GB
MAC Layer                     ~2.5 GB
──────────────────────────────────────
Used                          ~10.5 GB
Available for activations     ~13.5 GB  ✅
```

**结论：** 优化后的配置为激活值和梯度留出了充足的空间，可以安全运行。

---

## 🔬 对实验的影响

### num_memory_tokens 的作用：

`num_memory_tokens` 控制从 H-UAV 神经记忆中检索的 token 数量：

- **128 tokens**: 更丰富的记忆表示，但占用 8GB 显存
- **32 tokens**: 更紧凑的记忆表示，仅占用 2GB 显存

### 预期影响：

✅ **正面影响：**
- 显存占用减少 71%，解决 OOM 问题
- 训练和推理速度可能提升（更少的参数）
- 可以使用更大的 batch size

⚠️ **可能的影响：**
- 记忆表示能力略有下降（从 128 降到 32 tokens）
- 但对于初步实验，32 tokens 应该足够验证记忆机制的有效性

### 如果需要更激进的优化：

如果 2.5GB 仍然不够，可以进一步减小：

```python
# 方案 1: 进一步减少 num_memory_tokens (2.5GB → 0.9GB)
num_memory_tokens: int = 8

# 方案 2: 同时减小 memory_dim (2.5GB → 0.6GB)
memory_dim: int = 2048
num_memory_tokens: int = 8
```

---

## 📝 验证优化

运行以下命令验证配置：

```bash
# 1. 检查配置文件
grep "num_memory_tokens" hierarchical_uav/models/uav_config.py
grep "cache_size" hierarchical_uav/mac_memory/neural_memory.py

# 2. 分析内存占用
python analyze_mac_memory.py

# 3. 实际测试 H-UAV 启动
./run_huav.sh
```

---

## 🚀 下一步

现在您可以开始训练 H-UAV 的 MAC 记忆了：

```bash
# 1. 训练 H-UAV MAC 记忆
./train_huav.sh

# 2. 加载训练好的记忆启动 H-UAV
LOAD_MEMORY=./outputs/huav_training/huav_memory_final.pt ./run_huav.sh

# 3. 启动 L-UAV 测试记忆有效性
./run_luav.sh
```

预期训练的 MAC 记忆文件大小：~2.5 GB（之前是 ~9.1 GB）

---

## 📅 修改日期

2025-11-23

## ✍️ 修改人

Claude (AI Assistant)
