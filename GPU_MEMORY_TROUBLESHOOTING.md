# 🔧 GPU内存不足问题 - 完整解决方案

## 问题症状

当运行L-UAV时出现以下错误之一：
```
RuntimeError: CUDA out of memory
RuntimeError: CUDA error: out of memory
torch.cuda.OutOfMemoryError
```

---

## 🔍 原因分析

### 主要原因

1. **两个完整LLaVA模型占用内存过大**
   - H-UAV模型：约12-14GB
   - L-UAV模型：约12-14GB
   - 总计：24-28GB（超过大多数单卡容量）

2. **MAC内存模块额外开销**
   - Neural Memory: ~512MB
   - Persistent Memory: ~256MB
   - MAC Layer: ~1-2GB

3. **GPU内存分配不均**
   - 如果两张卡容量不同
   - 或H-UAV已占满GPU 0

4. **CUDA内存碎片化**
   - 长时间运行后内存碎片增加

---

## ✅ 解决方案（按推荐顺序）

### 方案1：使用8-bit量化（推荐，最有效）

**内存节省：约50%**

```bash
# 使用优化脚本（已配置8-bit）
./run_hierarchical_uav_optimized.sh h-uav  # Terminal 1
./run_hierarchical_uav_optimized.sh l-uav  # Terminal 2
```

**或手动运行：**
```bash
# H-UAV with 8-bit
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cuda:0 \
    --load_8bit \
    ...

# L-UAV with 8-bit
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device cuda:1 \
    --load_8bit \
    ...
```

**预期内存使用：**
- H-UAV：6-7GB（原12-14GB）
- L-UAV：6-7GB（原12-14GB）

---

### 方案2：减小MAC内存配置

**内存节省：约20-30%**

编辑 `run_hierarchical_uav_optimized.sh` 或直接使用：

```bash
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --memory_dim 2048 \           # 降低50%（原4096）
    --num_persistent_tokens 32 \   # 降低50%（原64）
    --num_memory_tokens 64 \       # 降低50%（原128）
    --load_8bit \
    ...
```

---

### 方案3：调整GPU分配

如果你的两个GPU容量不同：

```bash
# 方案3A：将两个模型都放在大内存GPU上（串行运行）
# Terminal 1 - 启动H-UAV
python hierarchical_uav/eval_task1.py --uav_type h-uav --device cuda:0 ...

# Terminal 2 - 等H-UAV评估完，再启动L-UAV
# (先 Ctrl+C 停止H-UAV)
python hierarchical_uav/eval_task1.py --uav_type l-uav --device cuda:0 ...

# 方案3B：将大模型放大卡，小模型放小卡
# 查看GPU容量
nvidia-smi --query-gpu=memory.total --format=csv,noheader

# 假设GPU 0有24GB，GPU 1有16GB
# H-UAV (更重，需要更新内存) -> GPU 0
# L-UAV (轻量，只推理) -> GPU 1
```

---

### 方案4：清理GPU缓存

```python
# 在运行前清理GPU缓存
python << EOF
import torch
torch.cuda.empty_cache()
print("GPU cache cleared")
EOF

# 然后运行
./run_hierarchical_uav_optimized.sh l-uav
```

---

### 方案5：使用梯度检查点（Gradient Checkpointing）

编辑 `hierarchical_uav/models/llava_mac.py`：

```python
# 在LLaVAWithMAC.__init__中添加
self.llava_model.config.use_cache = False
if hasattr(self.llava_model, 'gradient_checkpointing_enable'):
    self.llava_model.gradient_checkpointing_enable()
```

---

### 方案6：只在一个GPU上运行（最后手段）

如果只有一个大GPU或两个小GPU：

```bash
# 串行运行（不需要同时运行）
# 1. 先运行H-UAV收集数据
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cuda:0 \
    --load_8bit \
    --test_data data/metadata/test_small.jsonl

# 2. 保存H-UAV的内存状态
# (在代码中添加save_mac_state调用)

# 3. 停止H-UAV，运行L-UAV
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device cuda:0 \
    --load_8bit \
    --test_data data/metadata/test_small.jsonl
```

---

## 📊 内存使用对比表

| 配置 | H-UAV内存 | L-UAV内存 | 总计 | 适用GPU |
|------|-----------|-----------|------|---------|
| **默认配置** | 14GB | 14GB | 28GB | 2×16GB+ |
| **8-bit量化** | 7GB | 7GB | 14GB | 2×8GB+ |
| **8-bit + 减小配置** | 5GB | 5GB | 10GB | 2×6GB+ |
| **4-bit量化（实验）** | 4GB | 4GB | 8GB | 2×4GB+ |

---

## 🔍 诊断步骤

### 1. 检查GPU状态

```bash
# 运行诊断脚本
./check_gpu_memory.sh

# 或手动检查
nvidia-smi
```

查看：
- 两个GPU的总内存（Total Memory）
- 当前可用内存（Free Memory）
- H-UAV启动后GPU 0的使用量

### 2. 测试H-UAV内存使用

```bash
# Terminal 1 - 启动H-UAV
./run_hierarchical_uav_optimized.sh h-uav

# Terminal 2 - 监控GPU
watch -n 1 nvidia-smi
```

记录H-UAV稳定后GPU 0的内存使用量。

### 3. 计算L-UAV可用内存

```bash
# GPU 1总内存
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 1)

# 预留给系统
RESERVED=1024  # 1GB

# L-UAV可用
AVAILABLE=$((TOTAL - RESERVED))

echo "L-UAV可用内存: ${AVAILABLE}MB"
```

如果可用内存 < 6GB，必须使用8-bit量化。

---

## 💡 优化建议

### 短期（立即生效）

1. ✅ **使用优化脚本**：
   ```bash
   ./run_hierarchical_uav_optimized.sh h-uav
   ./run_hierarchical_uav_optimized.sh l-uav
   ```

2. ✅ **启用8-bit量化**：
   ```bash
   --load_8bit
   ```

3. ✅ **减小配置**：
   ```bash
   --memory_dim 2048 --num_persistent_tokens 32 --num_memory_tokens 64
   ```

### 中期（修改代码）

1. **实现模型权重共享**：
   - H-UAV和L-UAV共享LLaVA base model
   - 只在MAC层有差异
   - 预期节省：50%内存

2. **实现动态批处理**：
   - 减小batch size
   - 动态调整根据GPU状态

### 长期（架构优化）

1. **分布式推理**：
   - 使用DeepSpeed ZeRO
   - 模型并行

2. **混合精度训练**：
   - FP16/BF16
   - 自动混合精度

---

## 🧪 测试配置

### 测试1：最小配置（4GB GPU）

```bash
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --load_8bit \
    --memory_dim 1024 \
    --num_persistent_tokens 16 \
    --num_memory_tokens 32 \
    --test_data data/metadata/test_small.jsonl
```

### 测试2：平衡配置（8GB GPU）

```bash
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --load_8bit \
    --memory_dim 2048 \
    --num_persistent_tokens 32 \
    --num_memory_tokens 64
```

### 测试3：高性能配置（16GB+ GPU）

```bash
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --memory_dim 4096 \
    --num_persistent_tokens 64 \
    --num_memory_tokens 128
```

---

## 📝 常见问题 FAQ

### Q1: 8-bit量化会影响性能吗？

**A**: 轻微影响，准确率下降约1-2%，但内存节省50%+。对于实验来说完全可接受。

### Q2: 我只有1个GPU怎么办？

**A**: 使用串行运行方式（方案6），或使用模型共享（需要修改代码）。

### Q3: 两个GPU大小不同怎么分配？

**A**: 将H-UAV（需要更新内存）放在大GPU上，L-UAV（只推理）放在小GPU上。

### Q4: 还是内存不够怎么办？

**A**:
1. 使用4-bit量化（`--load_4bit`）
2. 减小测试集（只用10个样本）
3. 使用CPU（很慢，不推荐）

---

## 🆘 紧急救援

如果所有方案都失败：

```bash
# 1. 完全清理GPU
sudo nvidia-smi --gpu-reset

# 2. 重启Python环境
conda deactivate
conda activate airspatial

# 3. 使用最小配置
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cuda:0 \
    --load_8bit \
    --memory_dim 1024 \
    --num_persistent_tokens 16 \
    --num_memory_tokens 32 \
    --test_data data/metadata/test_small.jsonl

# 4. 如果还是失败，使用CPU
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cpu \
    --test_data data/metadata/test_small.jsonl
```

---

## 📞 获取帮助

如果问题仍未解决，提供以下信息：

```bash
# 1. GPU信息
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 2. PyTorch版本
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 3. 错误日志
cat outputs/hierarchical_uav/luav_optimized.log | tail -50

# 4. GPU使用情况
nvidia-smi
```

---

**记住：使用优化脚本是最简单的解决方案！**

```bash
./run_hierarchical_uav_optimized.sh h-uav
./run_hierarchical_uav_optimized.sh l-uav
```
