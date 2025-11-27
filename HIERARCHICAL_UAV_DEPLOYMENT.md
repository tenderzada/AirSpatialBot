# 层次化 UAV 系统部署指南

本指南介绍如何部署和运行基于 LoRA 记忆编织器的层次化 UAV 系统。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    H-UAV (记忆编织器)                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  冻结的 LLaVA (7B 参数)                            │  │
│  └───────────────────────────────────────────────────┘  │
│                          ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  LoRA 注入 @ 第 8, 16, 24 层                       │  │
│  │  - Target: q_proj, v_proj                         │  │
│  │  - 可训练参数: 8.4M (0.12%)                        │  │
│  └───────────────────────────────────────────────────┘  │
│                          ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  记忆 Token 生成器                                 │  │
│  │  输出: [8, 4096] 记忆序列                          │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
              REST API: POST /get_memory
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  L-UAV (轻量级推理)                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │  基础推理 → 置信度评估                             │  │
│  └───────────────────────────────────────────────────┘  │
│                          ↓                               │
│           置信度 < 阈值? → 请求 H-UAV 记忆                │
│                          ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  增强推理: memory || input → 最终答案              │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 训练 LoRA 记忆编织器

首先训练 H-UAV 的 LoRA 适配器：

```bash
./train_lora_injection_sqa.sh
```

**训练完成后会生成：**
- `./outputs/lora_injection_sqa/lora_adapters_final.pt` - 最终 LoRA 权重 (~34 MB)
- `./outputs/lora_injection_sqa/training_report.json` - 训练报告

### 2. 评估 H-UAV Standalone（测试 MemGen 效果）

在启动完整的协作系统之前，建议先测试 H-UAV 的 LoRA 记忆编织器效果：

```bash
./start_huav_standalone.sh
```

**用途：**
- 直接测试 LoRA 注入的效果
- 验证 MemGen 记忆编织器在 SQA 上的性能
- 作为上限性能参考

**输出：**
- `./outputs/eval_huav_standalone/results.json`
- 包含整体指标和分类型指标

### 3. 启动 H-UAV 服务器

在一个终端中启动 H-UAV LoRA 记忆服务器：

```bash
./start_huav_lora.sh
```

**服务器信息：**
- 端口: `8000`
- API 端点: `POST http://localhost:8000/get_memory`
- 输入: 图像 (base64) + 问题文本
- 输出: 记忆 tokens `[8, 4096]`

**检查服务器是否运行：**
```bash
curl http://localhost:8000/health
```

### 3. 运行 L-UAV 评估

#### 选项 A: L-UAV 协作模式（使用 H-UAV 记忆）

在另一个终端中启动 L-UAV，连接到 H-UAV：

```bash
./start_luav_with_huav.sh
```

**特性：**
- L-UAV 首先进行基础推理
- 当置信度 < 0.5 时，向 H-UAV 请求记忆
- 使用记忆进行增强推理
- 测量 H-UAV 使用率和性能提升

#### 选项 B: L-UAV 独立模式（基线对比）

运行纯 L-UAV 推理，不使用 H-UAV：

```bash
./start_luav_standalone.sh
```

**用途：**
- 建立基线性能
- 与协作模式对比，评估 H-UAV 记忆的效果

## 评估结果

### 输出文件

**协作模式：**
- `./outputs/eval_luav_with_huav/results.json`

**独立模式：**
- `./outputs/eval_luav_standalone/results.json`

### 结果格式

```json
{
  "metrics": {
    "mae": 123.45,    // 平均绝对误差
    "rmse": 234.56,   // 均方根误差
    "mre": 0.15       // 平均相对误差
  },
  "stats": {
    "total_queries": 1000,
    "huav_requests": 350,          // 请求 H-UAV 的次数
    "standalone_inferences": 650   // 独立推理的次数
  },
  "results": [...]
}
```

### 查看结果

```bash
# 查看协作模式结果
cat ./outputs/eval_luav_with_huav/results.json | jq '.metrics'

# 查看独立模式结果
cat ./outputs/eval_luav_standalone/results.json | jq '.metrics'

# 对比 H-UAV 使用率
cat ./outputs/eval_luav_with_huav/results.json | jq '.stats'
```

## 配置说明

### H-UAV 配置 (`start_huav_lora.sh`)

```bash
MODEL_PATH="/mnt/data/AirSpatialBot"              # 基础 LLaVA 模型
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"  # Vision encoder
LORA_WEIGHTS="./outputs/lora_injection_sqa/lora_adapters_final.pt"
PORT=8000
DEVICE="cuda:0"
TARGET_LAYERS="8 16 24"
LORA_RANK=8
```

### L-UAV 配置 (`start_luav_with_huav.sh`)

```bash
MODEL_PATH="/mnt/data/AirSpatialBot"              # L-UAV 模型
DEVICE="cuda:1"                                   # 使用不同 GPU
HUAV_URL="http://localhost:8000"                  # H-UAV 服务器地址
CONFIDENCE_THRESHOLD=0.5                          # 置信度阈值
TEST_DATA="/mnt/data/AirSpatial/airspatial_sqa_test.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
```

## 核心设计原则

### 1. 记忆编织器（H-UAV）

**不是**独立的多层网络，而是：
- LoRA 适配器注入到**冻结的** LLaVA
- 仅在特定层 (8, 16, 24) 注入
- 仅在特定模块 (q_proj, v_proj) 注入
- 基础 LLaVA **保持冻结**

**参数效率：**
- 基础 LLaVA: 7B 参数（冻结）
- LoRA 适配器: 8.4M 参数（可训练）
- 可训练参数占比: **0.12%**

### 2. 记忆序列

**H-UAV 返回的是记忆 tokens，不是答案！**

```python
# ✅ 正确
response = {
    'memory_tokens': [[...], ...],  # Shape: [8, 4096]
    'memory_shape': [8, 4096]
}

# ❌ 错误
response = {
    'answer': "123.45 meters",
    'confidence': 0.8
}
```

### 3. L-UAV 工作流

```
1. 基础推理 → 获得初步答案 + 置信度
2. 置信度检查:
   - 如果 confidence >= threshold → 使用基础答案
   - 如果 confidence < threshold → 请求 H-UAV 记忆
3. 增强推理: memory || input → 最终答案
```

## 技术细节

### LoRA 注入

```python
# 单个 LoRA 适配器
class LoRALinear(nn.Module):
    def forward(self, x, original_output):
        # y = W_0 @ x + (B @ A) @ x * (alpha / rank)
        lora_output = (x @ self.lora_A) @ self.lora_B
        return original_output + lora_output * self.scaling
```

**为什么选择层 8, 16, 24？**
- 层 8: 浅层，捕获低级视觉特征
- 层 16: 中层，捕获语义特征
- 层 24: 深层，捕获高级抽象
- 均匀分布，覆盖不同抽象层次

**为什么选择 q_proj, v_proj？**
- 注意力机制的核心组件
- 直接影响信息检索和整合
- 对记忆生成最有效

### 训练过程

```python
# 仅优化 LoRA 参数
lora_params = memory_weaver.get_lora_parameters()
optimizer = AdamW(lora_params, lr=1e-4)

for epoch in range(num_epochs):
    for batch in train_data:
        # 基础 LLaVA 保持冻结
        memory_tokens = memory_weaver.generate_memory_tokens(hidden_states)
        loss = mse_loss(memory_tokens, target)
        loss.backward()  # 只更新 LoRA 参数
        optimizer.step()
```

## 故障排除

### H-UAV 服务器无法启动

**检查 LoRA 权重：**
```bash
ls -lh ./outputs/lora_injection_sqa/lora_adapters_final.pt
```

如果文件不存在，先训练模型：
```bash
./train_lora_injection_sqa.sh
```

### L-UAV 无法连接到 H-UAV

**检查 H-UAV 服务器状态：**
```bash
curl http://localhost:8000/health
```

**检查端口是否被占用：**
```bash
netstat -tuln | grep 8000
```

**防火墙问题：**
如果 H-UAV 和 L-UAV 在不同机器上，确保端口 8000 开放。

### 显存不足

**使用 8-bit 量化：**
所有脚本已默认启用 `--load_8bit`

**降低 batch size：**
编辑 `train_lora_injection_sqa.sh`:
```bash
BATCH_SIZE=2  # 从 4 降到 2
```

**使用更小的 LoRA rank：**
```bash
LORA_RANK=4  # 从 8 降到 4
```

## 性能基准

### 预期结果

**L-UAV 独立模式（基线）：**
- MAE: ~200-300
- RMSE: ~300-400
- H-UAV 使用率: 0%
- 说明：纯轻量级推理，无记忆增强

**H-UAV Standalone（MemGen 上限）：**
- MAE: 预期 < L-UAV 独立模式 20-30%
- RMSE: 预期 < L-UAV 独立模式 20-30%
- 说明：直接使用 LoRA 记忆编织器，展示 MemGen 最佳性能

**L-UAV 协作模式（实际部署）：**
- MAE: 预期介于 L-UAV 独立和 H-UAV standalone 之间
- RMSE: 预期介于 L-UAV 独立和 H-UAV standalone 之间
- H-UAV 使用率: 30-50%
- 说明：自适应协作，平衡性能和效率

**性能关系：**
```
L-UAV standalone (基线)
    ↓ (性能提升)
L-UAV + H-UAV 协作 (实际部署)
    ↓ (接近上限)
H-UAV standalone (MemGen 上限)
```

**关键指标：**
- H-UAV 请求率应该在 30-50% 之间（说明 self-matching 机制工作正常）
- 协作模式的误差应该明显低于独立模式
- H-UAV standalone 展示 LoRA 记忆编织器的最佳效果

## 进阶使用

### 调整置信度阈值

编辑 `start_luav_with_huav.sh`:
```bash
CONFIDENCE_THRESHOLD=0.3  # 更激进地使用 H-UAV
CONFIDENCE_THRESHOLD=0.7  # 更保守地使用 H-UAV
```

### 限制测试样本数量（快速测试）

编辑启动脚本:
```bash
MAX_SAMPLES="100"  # 只测试前 100 个样本
```

### 使用不同的 GPU

```bash
# H-UAV 在 GPU 0
DEVICE="cuda:0"  # in start_huav_lora.sh

# L-UAV 在 GPU 1
DEVICE="cuda:1"  # in start_luav_with_huav.sh
```

## 文件结构

```
AirSpatialBot/
├── train_lora_injection_sqa.sh          # 训练 LoRA 记忆编织器
├── start_huav_standalone.sh             # H-UAV 独立评估 (测试 MemGen)
├── start_huav_lora.sh                   # 启动 H-UAV 服务器
├── start_luav_with_huav.sh              # 启动 L-UAV (协作)
├── start_luav_standalone.sh             # 启动 L-UAV (独立)
├── hierarchical_uav/
│   ├── mac_memory/
│   │   └── lora_injection.py            # LoRA 注入实现
│   ├── train_lora_injection.py          # 训练脚本
│   ├── eval_huav_standalone.py          # H-UAV 独立评估
│   ├── huav_lora_server_v2.py          # H-UAV 服务器
│   └── eval_sqa_lora.py                # L-UAV 评估
└── outputs/
    ├── lora_injection_sqa/              # 训练输出
    │   ├── lora_adapters_final.pt
    │   └── training_report.json
    ├── eval_huav_standalone/            # H-UAV 独立评估结果
    │   └── results.json
    ├── eval_luav_with_huav/             # 协作模式结果
    │   └── results.json
    └── eval_luav_standalone/            # 独立模式结果
        └── results.json
```

## 引用

如果您使用本系统，请引用：

```bibtex
@article{memgen2024,
  title={MemGen: Memory is a Generative Model},
  author={...},
  journal={...},
  year={2024}
}
```

---

**完成状态：**
- ✅ LoRA 记忆编织器训练
- ✅ H-UAV 服务器部署
- ✅ L-UAV 评估脚本
- ✅ 协作机制实现

**下一步：**
1. 运行完整评估
2. 对比协作 vs 独立模式性能
3. 调优置信度阈值
4. 分析 H-UAV 使用模式
