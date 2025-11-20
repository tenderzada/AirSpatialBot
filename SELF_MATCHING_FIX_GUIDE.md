# 自匹配机制修复指南

## 📋 问题总结

您的评估结果发现了以下问题：

1. **自匹配分数异常** - 所有样本的分数都是 1.0（应该有分布）
2. **没有使用 H-UAV 通信** - 100% 本地决策，0% 远程查询
3. **缓存未使用** - 0% 缓存命中（因为没有远程查询）

## 🔍 根本原因分析

### 问题 1: 自匹配分数计算错误

**原代码问题**（`self_matching.py:94-155`）：

```python
# L2 归一化
query = F.normalize(query, dim=-1)  # 范数 = 1.0
key = F.normalize(key, dim=-1)      # 范数 = 1.0

# 但后续计算范数差异
query_magnitude = torch.norm(query, dim=-1)  # 总是 ≈ 1.0
key_magnitude = torch.norm(key, dim=-1)      # 总是 ≈ 1.0
magnitude_diff = |1.0 - 1.0| = 0
scores = 1.0 / (1.0 + 0) = 1.0  # ❌ 恒定为 1.0！
```

**修复方案**：

```python
# 1. 添加可学习的投影层
self.key_projection = nn.Linear(key_dim, query_dim)

# 2. 使用余弦相似度
key_projected = self.key_projection(key)
key_projected_norm = F.normalize(key_projected, dim=-1)
query_norm = F.normalize(query, dim=-1)

# 3. 计算点积（余弦相似度）
scores = (query_norm * key_projected_norm).sum(dim=-1)
scores = (scores + 1.0) / 2.0  # 映射到 [0, 1]
```

### 问题 2: 使用随机特征

**原代码问题**（`eval_task1.py:159-160`）：

```python
image_features = torch.randn(...)  # ❌ 随机特征
bbox_3d = torch.randn(...)         # ❌ 随机bbox
```

**修复方案**：

- 添加 `extract_image_features()` - 从 LLaVA vision tower 提取真实特征
- 添加 `parse_bbox_3d()` - 从样本数据解析 3D 边界框
- 添加 `--image_dir` 参数指定图像目录

---

## ✅ 已实现的修复

### 1. 自匹配分数计算修复

**文件**: `hierarchical_uav/communication/self_matching.py`

**核心修改**：

- ✅ 添加可学习的 key→query 投影层
- ✅ 使用余弦相似度替代范数差异
- ✅ 分数范围：[0, 1]，低分 = 不确定 = 查询 H-UAV

**新增功能**：

```python
# 获取分数统计
stats = sm_module.gate.get_score_statistics()
# 返回: {'mean': float, 'std': float, 'min': float, 'max': float}

# 自适应阈值调整
new_threshold = sm_module.auto_adjust_threshold(target_query_rate=0.3)
# 自动调整阈值以达到 30% 查询率
```

### 2. 特征提取改进

**文件**: `hierarchical_uav/eval_task1.py`

**新增功能**：

```python
# 从真实图像提取特征
def extract_image_features(model, image_path, bbox_3d):
    # 1. 加载图像
    # 2. 通过 vision tower 提取特征
    # 3. 通过 mm_projector 投影
    # 4. 返回特征向量

# 解析 3D 边界框
def parse_bbox_3d(sample):
    # 支持 3D bbox、2D bbox，或使用默认值
```

**新增参数**：

```bash
--image_dir ./data/images    # 图像目录路径
```

### 3. 统计输出改进

**新增输出**：

```
Self-matching score distribution:
  Mean: 0.6234
  Std:  0.1456
  Min:  0.2145
  Max:  0.9876

Score distribution by range:
  [0.0, 0.3):   45 ( 4.6%)
  [0.3, 0.5):  123 (12.7%)
  [0.5, 0.7):  456 (47.0%)
  [0.7, 0.9):  298 (30.7%)
  [0.9, 1.0):   49 ( 5.0%)
```

---

## 🚀 使用修复后的代码

### 快速测试

运行测试脚本验证修复：

```bash
python test_self_matching_fix.py
```

**预期输出**：

```
✓ PASSED: Scores show variation (std=0.xxxx)
✓ PASSED: Scores in valid range [0, 1]
✓ PASSED: H-UAV queries triggered (xx.x% query rate)
✓ PASSED: Query/key dimensions correct
```

### 重新运行评估

使用相同的命令，但现在应该看到正常的分数分布：

```bash
# Terminal 1 - H-UAV
./run_hierarchical_uav_optimized.sh h-uav

# Terminal 2 - L-UAV
./run_hierarchical_uav_optimized.sh l-uav
```

**如果有图像数据**，指定图像目录：

```bash
python hierarchical_uav/eval_task1.py \
  --uav_type l-uav \
  --device cuda:1 \
  --image_dir ./data/images \
  --load_8bit
```

---

## 📊 预期改进

### 修复前（您的报告）

```
自匹配分数: 全部 1.0
标准差: 0.0000
本地决策: 971 (100.0%)
远程查询: 0 (0.0%)
```

### 修复后（预期）

```
自匹配分数: 合理分布（例如 0.3 - 0.9）
标准差: 0.10 - 0.20
本地决策: 600-700 (60-70%)
远程查询: 200-300 (20-30%)
缓存命中: 开始累积
```

---

## 🔧 阈值调整建议

### 默认阈值

当前默认: `threshold=0.7`

**解释**：
- 分数 < 0.7 → 查询 H-UAV（不确定）
- 分数 ≥ 0.7 → 本地处理（有信心）

### 调整策略

1. **提高查询率**（更保守，更准确）：

```bash
--threshold 0.8  # 80%+ 相似度才本地处理
```

2. **降低查询率**（更激进，更快速）：

```bash
--threshold 0.5  # 50%+ 相似度即本地处理
```

3. **自适应调整**（代码中）：

```python
# 在评估循环中
if i > 0 and i % 100 == 0:  # 每 100 个样本
    new_threshold = sm_module.auto_adjust_threshold(target_query_rate=0.3)
    print(f"Adjusted threshold to {new_threshold:.4f}")
```

### 推荐设置

| 场景 | 阈值 | 查询率 | 特点 |
|------|------|--------|------|
| **高准确度优先** | 0.8 | ~40% | 更多查询 H-UAV，更准确 |
| **平衡（推荐）** | 0.7 | ~30% | 带宽与性能平衡 |
| **低延迟优先** | 0.6 | ~20% | 更多本地处理，更快 |
| **极致速度** | 0.5 | ~10% | 几乎全本地，可能牺牲准确度 |

---

## 🐛 故障排查

### 问题 1: 分数仍然全是 1.0

**检查**：
- 是否拉取了最新代码？`git pull`
- 是否重新加载了模块？重启 Python 进程

### 问题 2: 仍然 0% 远程查询

**检查**：
- 查看分数分布，如果 min=0.9, max=1.0，说明特征太相似
- 尝试降低阈值：`--threshold 0.95`
- 确认 H-UAV 服务器正在运行

### 问题 3: "Image not found" 警告

**原因**：图像路径不正确

**解决**：
- 检查 `--image_dir` 参数
- 查看样本中的 `image_id` 字段格式
- 可以暂时忽略，代码会自动降级到随机特征

### 问题 4: 查询率过高或过低

**解决**：
- 过高（>50%）：降低阈值 `--threshold 0.6`
- 过低（<10%）：提高阈值 `--threshold 0.8`
- 或使用自适应调整（见上文）

---

## 📝 技术细节

### 自匹配分数含义

**公式**（修复后）：

```
m_{i,i} = cosine_similarity(μ_i, π(κ_i))
        = (μ_i^T · π(κ_i)) / (||μ_i|| ||π(κ_i)||)
```

其中：
- `μ_i`: query 向量（256 维）
- `κ_i`: key 向量（1024 维）
- `π(·)`: 可学习的投影层（1024 → 256）

**分数解释**：
- `1.0`: query 和 key 完全一致 → 非常确定 → 本地处理
- `0.5`: query 和 key 中等相似 → 不确定 → 可能查询
- `0.0`: query 和 key 完全相反 → 非常不确定 → 查询 H-UAV

### 投影层为什么重要

**问题**：query (256D) 和 key (1024D) 维度不同，无法直接计算相似度

**解决**：
1. 添加线性投影层：`Linear(1024 → 256)`
2. 将 key 投影到 query 空间
3. 在同一空间计算余弦相似度

**好处**：
- 可学习：投影层参数可以训练优化
- 语义对齐：确保 query 和 key 在同一语义空间比较
- 灵活性：可以轻松改变维度

---

## 📚 相关文件

修改的文件：
- ✅ `hierarchical_uav/communication/self_matching.py` - 核心修复
- ✅ `hierarchical_uav/eval_task1.py` - 特征提取改进

新增文件：
- ✅ `test_self_matching_fix.py` - 测试脚本
- ✅ `SELF_MATCHING_FIX_GUIDE.md` - 本文档

---

## 🎯 下一步

1. ✅ **验证修复** - 运行 `test_self_matching_fix.py`
2. ✅ **重新评估** - 使用修复后的代码运行评估
3. ✅ **检查分数分布** - 确认不再是全 1.0
4. ✅ **观察查询率** - 应该在 20-40% 范围
5. ⚙️ **调整阈值**（可选）- 根据需要优化查询率
6. 📊 **分析结果** - 对比修复前后的性能

---

## 💡 关键要点

1. **自匹配分数现在有意义了** - 反映 query/key 的真实相似度
2. **会产生实际的 H-UAV 查询** - 根据阈值和分数分布
3. **使用真实图像特征**（如果提供图像）- 更准确的决策
4. **支持自适应阈值** - 可以动态调整查询率
5. **详细的统计输出** - 便于分析和调试

---

## 📞 需要帮助？

如果遇到问题：

1. 运行诊断脚本：`python test_self_matching_fix.py`
2. 检查分数统计输出
3. 调整阈值参数
4. 查看本文档的"故障排查"部分

祝评估顺利！🚀
