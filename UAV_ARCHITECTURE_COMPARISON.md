# H-UAV vs L-UAV 架构对比

## 核心问题

1. **都加载了LLaVA模型吗？** ✅ 是的，完全相同的基础模型
2. **L-UAV基于LLaVA做了什么？** → 添加记忆注入能力
3. **H-UAV基于LLaVA做了什么？** → 添加记忆存储和学习能力

---

## 详细对比表

| 维度 | H-UAV | L-UAV |
|------|-------|-------|
| **基础模型** | LLaVA-v1.5 | LLaVA-v1.5（相同） |
| **视觉编码器** | CLIP ViT-L/14@336 | CLIP ViT-L/14@336（相同） |
| **MAC记忆层** | ✅ 有，且可更新 | ❌ 没有（或冻结） |
| **参数状态** | MAC层可训练 | 全部冻结 |
| **记忆能力** | 存储+检索+更新 | 接收+注入 |
| **通信角色** | gRPC服务器（被动） | gRPC客户端（主动） |
| **资源消耗** | 高（GPU 0，更新记忆） | 低（GPU 1，仅推理） |
| **独特组件** | MACLayer | memory_to_visual_proj + self_matching |
| **主要功能** | 积累知识 | 借用知识 |

---

## 架构细节

### 1. 共同的基础（两者都有）

```python
# hierarchical_uav/models/llava_mac.py:44-86

class LLaVAWithMAC(nn.Module):
    def __init__(self, config: UAVConfig):
        # ===== 两者都执行这部分 =====

        # 1. 加载相同的LLaVA模型
        self.tokenizer, self.llava_model, self.image_processor, _ = \
            load_pretrained_model(
                model_path="/mnt/data/AirSpatialBot",      # 相同
                vision_tower="clip-vit-large-patch14-336",  # 相同
                load_8bit=True                              # 相同
            )

        # 2. LLaVA包含的组件：
        #    - Vision Tower: CLIP ViT-L/14 (视觉编码器)
        #    - mm_projector: 2层MLP (视觉-文本投影)
        #    - LLaMA: 语言模型主干
```

**LLaVA基础架构：**
```
输入图像 [3, 336, 336]
    ↓
CLIP Vision Encoder
    ↓
Vision Features [576, 1024]
    ↓
mm_projector (2-layer MLP)
    ↓
Visual Tokens [576, 4096]  ← 这里是H-UAV和L-UAV的分叉点
```

---

### 2. H-UAV的扩展

```python
# hierarchical_uav/models/llava_mac.py:103-125, 135-144

# ===== H-UAV特有配置 =====

def _add_mac_layers(self):
    """在视觉特征之后添加MAC记忆层"""

    # MAC层配置
    self.mac_layer = MACLayer(
        hidden_size=4096,              # 匹配LLaVA
        num_persistent_tokens=64,      # 长期记忆（不丢失）
        num_memory_tokens=128,         # 工作记忆（可替换）
        enable_memory_update=True      # H-UAV独有：允许更新
    )

def _configure_huav(self):
    """H-UAV配置：允许学习"""
    self.mac_layer.unfreeze_for_huav()  # 解冻MAC参数
```

**H-UAV完整流程：**
```
图像 [3, 336, 336]
    ↓
CLIP Vision Encoder (冻结)
    ↓
Vision Features [576, 1024]
    ↓
mm_projector (冻结)
    ↓
Visual Tokens [576, 4096]
    ↓
┌─────────────────────────────┐
│  MAC Layer (可学习)          │
│  - Persistent Memory [64]   │  ← test-time learning在这里发生
│  - Working Memory [128]     │  ← 根据surprise和relevance更新
│  - Cross-Attention          │
└─────────────────────────────┘
    ↓
Enhanced Visual Tokens [576, 4096]
    ↓
LLaMA Language Model
    ↓
生成答案

同时：MAC层的记忆可以被L-UAV查询
```

**MAC记忆更新机制：**
```python
# hierarchical_uav/mac_memory/mac_layer.py

# 每次推理时：
surprise = compute_surprise(current_features, memory)
if surprise > threshold:
    # 这个样本很新颖，值得记住
    memory = update_memory(
        memory=memory,
        new_features=current_features,
        surprise=surprise,
        learning_rate=theta,
        forgetting_rate=alpha
    )
```

**gRPC服务：**
```python
# hierarchical_uav/communication/server.py

class HUAVServer:
    def QueryMemory(self, request):
        # 1. 接收L-UAV的查询向量
        query_vector = request.query  # [256]

        # 2. 从MAC记忆中检索
        memory_value = self.mac_layer.retrieve(query_vector)  # [4096]

        # 3. 返回给L-UAV
        return MemoryResponse(value=memory_value)
```

---

### 3. L-UAV的扩展

```python
# hierarchical_uav/models/llava_mac.py:129-133, 146-155, 264-430

# ===== L-UAV特有配置 =====

def _add_mac_layers(self):
    """添加记忆投影层（不是完整的MAC）"""

    # 投影层：将H-UAV的记忆转换为视觉token
    self.memory_to_visual_proj = nn.Linear(
        in_features=4096,   # H-UAV记忆维度
        out_features=4096   # LLaVA视觉token维度
    )

def _configure_luav(self):
    """L-UAV配置：只推理"""
    for param in self.parameters():
        param.requires_grad = False  # 冻结所有参数
```

**L-UAV完整流程：**

**阶段1：自匹配决策**
```python
# hierarchical_uav/communication/self_matching.py

图像特征 → Self-Matching Module
                ↓
    计算自相似度分数 (cosine similarity)
                ↓
        score < 0.7?
         ↙        ↘
       是          否
    查询H-UAV    本地推理
       ↓
  获取记忆特征
```

**阶段2：记忆注入推理**
```python
# hierarchical_uav/models/llava_mac.py:264-430

def generate_with_memory(images, memory_features, memory_weight):
    # 1. 提取图像特征
    image_features = vision_tower(images)      # [1, 576, 1024]
    image_features = mm_projector(image_features)  # [1, 576, 4096]

    # 2. 投影H-UAV记忆为视觉token
    memory_tokens = memory_to_visual_proj(memory_features)  # [1, 4096]
    memory_tokens = memory_tokens.unsqueeze(1)              # [1, 1, 4096]

    # 3. 应用动态权重（Phase 1优化）
    from dynamic_memory_weight import DynamicMemoryWeightAdjuster
    adjuster = DynamicMemoryWeightAdjuster(base_weight=0.5)
    weight = adjuster.compute_weight(self_match_score)
    # 如果score=0.47 → weight=0.3（低置信度）
    # 如果score=0.55 → weight=0.5（中等置信度）
    memory_tokens = memory_tokens * weight

    # 4. 拼接图像特征和记忆token
    augmented_features = torch.cat([
        image_features,   # [1, 576, 4096] - 原始图像
        memory_tokens     # [1, 1, 4096]   - H-UAV记忆
    ], dim=1)            # [1, 577, 4096] - 增强特征

    # 5. 手动构建inputs_embeds（避免LLaVA重复编码）
    text_embeds = embed_tokens(input_ids)
    final_embeds = insert_image_features(text_embeds, augmented_features)

    # 6. 生成答案
    return llama_model.generate(inputs_embeds=final_embeds)
```

**可视化L-UAV推理：**
```
样本: "这辆车是什么颜色？"
    ↓
提取图像特征 → Self-matching score = 0.47 (低于0.7)
    ↓
查询H-UAV (gRPC)
    ↓
收到记忆向量 [4096]
    ↓
┌──────────────────────────────────────┐
│  记忆注入过程                          │
│                                       │
│  图像token [576个]                    │
│  ┌─┬─┬─┬───┬─┬─┐                     │
│  │•│•│•│...│•│•│  原始视觉理解        │
│  └─┴─┴─┴───┴─┴─┘                     │
│        +                              │
│  记忆token [1个]                      │
│  ┌─────────┐                          │
│  │ ★ │  H-UAV的"经验"（权重0.3）     │
│  └─────────┘                          │
│        ↓                              │
│  合并 [577个token]                    │
│  ┌─┬─┬─┬───┬─┬─┬─────┐              │
│  │•│•│•│...│•│•│ ★ │                │
│  └─┴─┴─┴───┴─┴─┴─────┘              │
└──────────────────────────────────────┘
    ↓
LLaMA生成："这辆车是蓝色的"
```

---

## 关键实现对比

### Forward Pass（训练/推理）

**H-UAV:**
```python
def forward(input_ids, images, update_memory=True):
    # 1. 视觉编码
    image_features = vision_tower(images)
    image_features = mm_projector(image_features)

    # 2. MAC层增强 + 记忆更新
    image_features, memory_metrics = mac_layer(
        image_features,
        update_memory=True  # ← H-UAV在推理时也更新记忆
    )

    # 3. LLaVA生成
    outputs = llava_model(
        input_ids=input_ids,
        images=image_features
    )

    return outputs, memory_metrics
```

**L-UAV:**
```python
def forward(input_ids, images):
    # L-UAV不使用forward，只用generate_with_memory
    # 所有参数冻结，纯推理模式
    pass
```

### Generation（生成）

**H-UAV:**
```python
@torch.no_grad()
def generate(input_ids, images):
    # 标准生成，带MAC增强
    image_features = vision_tower(images)
    image_features = mm_projector(image_features)
    image_features, _ = mac_layer(image_features, update_memory=False)

    return llava_model.generate(
        input_ids=input_ids,
        images=image_features
    )
```

**L-UAV:**
```python
@torch.no_grad()
def generate_with_memory(input_ids, images, memory_features, memory_weight):
    # 关键：注入H-UAV的记忆
    image_features = vision_tower(images)
    image_features = mm_projector(image_features)

    # 投影并拼接记忆
    memory_tokens = memory_to_visual_proj(memory_features)
    memory_tokens = memory_tokens * memory_weight  # 动态权重
    augmented = torch.cat([image_features, memory_tokens], dim=1)

    # 手动构建inputs_embeds（绕过LLaVA的图像编码）
    text_embeds = embed_tokens(input_ids)
    final_embeds = insert_image_at_token(text_embeds, augmented)

    return llama_model.generate(inputs_embeds=final_embeds)
```

---

## 参数统计

### H-UAV参数

```
LLaVA Base (冻结):
  - CLIP ViT-L: 304M
  - mm_projector: 8M
  - LLaMA-7B: 7B

MAC Layer (可训练):
  - Persistent Memory: 64 × 4096 = 262K
  - Working Memory: 128 × 4096 = 524K
  - Attention weights: ~2M

Total Trainable: ~2.8M (仅MAC)
Total Frozen: ~7.3B (LLaVA)
```

### L-UAV参数

```
LLaVA Base (冻结):
  - 与H-UAV完全相同: 7.3B

Memory Projection (冻结):
  - Linear(4096 → 4096): 16M

Self-Matching Module (冻结):
  - Query MLP: ~1M
  - Key MLP: ~1M

Total Trainable: 0 (全部冻结)
Total Parameters: ~7.3B (仅推理)
```

---

## 通信协议

### gRPC接口定义

```protobuf
// hierarchical_uav/communication/proto/uav_service.proto

service UAVService {
  // L-UAV查询H-UAV记忆
  rpc QueryMemory(MemoryRequest) returns (MemoryResponse);

  // 健康检查
  rpc Ping(PingRequest) returns (PingResponse);
}

message MemoryRequest {
  repeated float query = 1;  // [256] 查询向量
  repeated float bbox = 2;   // [7] 3D边界框
}

message MemoryResponse {
  repeated float value = 1;  // [4096] 记忆特征
  bool cache_hit = 2;        // 是否命中缓存
}
```

### 通信流程

```
L-UAV                           H-UAV
  │                               │
  │  1. Self-matching             │
  │     score = 0.47 < 0.7        │
  │     决定查询H-UAV             │
  │                               │
  │  2. gRPC Request              │
  │  ──────────────────────────>  │
  │  query=[256], bbox=[7]        │
  │                               │
  │                               │  3. 检索MAC记忆
  │                               │     similarity = cosine(query, memory_keys)
  │                               │     value = weighted_sum(memory_values)
  │                               │
  │  4. gRPC Response             │
  │  <──────────────────────────  │
  │  value=[4096], cache_hit=True │
  │                               │
  │  5. 注入记忆推理              │
  │     augmented_features =      │
  │       [image | memory]        │
  │                               │
  │  6. 生成答案                  │
  │     "蓝色"                    │
  │                               │
```

---

## 实际运行示例

### H-UAV启动

```bash
./run_huav.sh

# 输出:
============================================================
Starting H-UAV (High Resource UAV)
============================================================
Configuration:
  Model: /mnt/data/AirSpatialBot
  Vision Tower: /mnt/data/clip-vit-large-patch14-336
  Device: cuda:0
  Port: 50051

Loading base LLaVA model...
✓ Base LLaVA model loaded
✓ MAC layers added
✓ Configured as H-UAV (learning mode)

Starting gRPC server on port 50051...
✓ H-UAV Server started

Processing samples:
  Sample 1: surprise=0.85 → 更新记忆
  Sample 2: surprise=0.23 → 跳过更新
  ...
```

### L-UAV启动

```bash
# 编辑 run_luav.sh:
FORCE_QUERY_RATE=0.3  # 研究模式：强制30%查询

./run_luav.sh

# 输出:
============================================================
Starting L-UAV (Low Resource UAV)
============================================================
Configuration:
  Device: cuda:1
  H-UAV Address: localhost:50051
  Self-matching Threshold: 0.7
  Force Query Rate: 0.3

⚠️  RESEARCH MODE: Force query rate = 30.0%
   → 30.0% of samples will bypass KB and test H-UAV memory

Connecting to H-UAV at localhost:50051...
✓ H-UAV server is reachable

Loading L-UAV model on cuda:1...
✓ Base LLaVA model loaded
✓ Configured as L-UAV (inference-only mode)

Evaluating on 971 samples:
  Sample 1: score=0.48 → Query H-UAV (weight=0.3)
  Sample 2: score=0.92 → Local inference
  Sample 3: score=0.51 → Query H-UAV (weight=0.5)
  ...

============================================================
L-UAV Evaluation Statistics
============================================================
Total samples: 971
Local decisions: 650 (67.0%)
Remote queries: 321 (33.0%)
  ↳ Forced queries (research mode): 291 (30.0%)
  ↳ Natural queries (self-matching): 30 (3.1%)
```

---

## 总结：核心差异

### H-UAV的使命
- **角色**: 知识银行 + 记忆管理者
- **能力**: 存储、学习、检索
- **特点**: 高资源消耗，支持test-time learning
- **类比**: 老师（有丰富经验，能不断学习新知识）

### L-UAV的使命
- **角色**: 轻量推理者 + 知识借用者
- **能力**: 快速推理 + 按需查询
- **特点**: 低资源消耗，纯推理模式
- **类比**: 学生（资源有限，遇到难题问老师）

### 协作模式
```
L-UAV: "这道题我不太确定..."
       (self-match score = 0.47 < 0.7)

L-UAV → H-UAV: "能帮我看看吗？"
                (gRPC query)

H-UAV: "我记得类似的情况..."
       (检索MAC记忆)

H-UAV → L-UAV: "这是我的经验"
                (返回memory_features [4096])

L-UAV: "结合你的经验和我的理解..."
       (augmented_features = [image | memory])

L-UAV: "答案是蓝色！"
       (generate with memory)
```

---

## 文件位置参考

- **模型定义**: `hierarchical_uav/models/llava_mac.py`
- **配置**: `hierarchical_uav/models/uav_config.py`
- **MAC记忆**: `hierarchical_uav/mac_memory/mac_layer.py`
- **自匹配**: `hierarchical_uav/communication/self_matching.py`
- **gRPC服务**: `hierarchical_uav/communication/server.py`
- **gRPC客户端**: `hierarchical_uav/communication/client.py`
- **评估脚本**: `hierarchical_uav/eval_task1.py`
- **动态权重**: `hierarchical_uav/dynamic_memory_weight.py`
