# 🚀 Hierarchical UAV System - Quick Start Guide

## 最小化开始（5分钟）

### 1. 准备环境

```bash
# 确保已安装LLaVA
./install_llava.sh

# 确保数据已准备
./setup_data.sh
```

### 2. 测试单个组件

```bash
# 测试MAC记忆模块
python -c "from hierarchical_uav.mac_memory import NeuralMemory; print('✓ MAC Memory OK')"

# 测试通信模块
python -c "from hierarchical_uav.communication import SelfMatchingGate; print('✓ Communication OK')"

# 测试模型配置
python -c "from hierarchical_uav.models import UAVConfig; print('✓ Models OK')"
```

### 3. 运行系统

**方式1：自动化运行（推荐用于演示）**

```bash
# 一键运行H-UAV和L-UAV
./run_hierarchical_uav.sh both
```

**方式2：手动运行（推荐用于调试）**

```bash
# Terminal 1 - 启动H-UAV服务器（GPU 0）
./run_hierarchical_uav.sh h-uav

# Terminal 2 - 启动L-UAV客户端（GPU 1）
# 等待H-UAV准备好后
./run_hierarchical_uav.sh l-uav
```

---

## 完整使用流程

### 步骤1：理解架构

```
     H-UAV (GPU 0)                    L-UAV (GPU 1)
┌──────────────────┐              ┌──────────────────┐
│  LLaVA + MAC     │              │  LLaVA + MAC     │
│  (Learning)      │◄─────────────│  (Frozen)        │
│                  │   Query μ    │                  │
│  Neural Memory   │──────────────►  Self-Matching   │
│  Episodic Cache  │   Value v    │  Gate (τ=0.7)   │
└──────────────────┘              └──────────────────┘
   Server:50051                      Client
```

### 步骤2：配置参数

编辑 `run_hierarchical_uav.sh` 修改配置：

```bash
# 模型路径
MODEL_PATH="./models/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"

# 数据路径
TEST_DATA="./data/metadata/airspatial_agent_test_task1.jsonl"

# 通信端口
PORT=50051

# 输出目录
OUTPUT_DIR="./outputs/hierarchical_uav"
```

### 步骤3：启动H-UAV

```bash
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --model_path ./models/AirSpatialBot \
    --vision_tower /mnt/data/clip-vit-large-patch14-336 \
    --device cuda:0 \
    --port 50051 \
    --test_data ./data/metadata/airspatial_agent_test_task1.jsonl
```

**预期输出：**
```
==========================================
Starting H-UAV Evaluation
==========================================

Loading base LLaVA model...
✓ Base LLaVA model loaded
✓ MAC layers added
✓ Configured as H-UAV (learning mode)

✓ H-UAV Server started
  Listening on port 50051
  Waiting for L-UAV connections...

Press Ctrl+C to stop server
```

### 步骤4：启动L-UAV

在另一个终端：

```bash
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --model_path ./models/AirSpatialBot \
    --vision_tower /mnt/data/clip-vit-large-patch14-336 \
    --device cuda:1 \
    --huav_address localhost:50051 \
    --threshold 0.7 \
    --test_data ./data/metadata/airspatial_agent_test_task1.jsonl
```

**预期输出：**
```
==========================================
Starting L-UAV Evaluation
==========================================

Connecting to H-UAV at localhost:50051...
✓ H-UAV server is reachable

Loading L-UAV model on cuda:1...
✓ L-UAV model loaded
✓ Self-matching module initialized (threshold=0.7)

Evaluating on 500 samples...
L-UAV Evaluation: 100%|████████| 500/500 [10:23<00:00]

==========================================
L-UAV Evaluation Statistics
==========================================
Total samples: 500
Local decisions: 320 (64.0%)
Remote queries: 180 (36.0%)

Self-matching statistics:
  Query rate: 36.00%
  Total queries: 500
  H-UAV queries: 180
```

---

## 关键参数说明

### Memory配置

```python
--memory_dim 4096           # 记忆向量维度
--memory_depth 2            # 记忆MLP深度
--num_persistent_tokens 64  # 持久记忆token数
--num_memory_tokens 128     # 检索记忆token数
```

### Learning配置（H-UAV）

```python
--surprise_eta 0.9      # 惊讶度衰减（momentum）
--learning_theta 0.1    # 学习率
--forgetting_alpha 0.01 # 遗忘率（weight decay）
```

### Communication配置（L-UAV）

```python
--threshold 0.7              # 自匹配阈值τ
--huav_address localhost:50051  # H-UAV地址
```

---

## 输出文件

```
outputs/hierarchical_uav/
├── huav_results.jsonl       # H-UAV服务统计
├── luav_results.jsonl       # L-UAV评估结果
├── huav.log                 # H-UAV日志
├── luav.log                 # L-UAV日志
└── memory_checkpoint.pt     # 记忆状态（可选）
```

### 结果格式

**luav_results.jsonl：**
```json
{
  "question_id": 1,
  "image_id": "image_001.jpg",
  "self_match_score": 0.65,
  "source": "huav",
  "cache_hit": false
}
```

---

## 性能指标

### 通信效率

- **Query Rate**: 20-40% 的样本需要查询H-UAV
- **Bandwidth**: 每个query ~1KB（256 floats）
- **Latency**: <100ms per query

### 记忆统计

- **Cache Hit Rate**: 随时间提升 (0% → 60%+)
- **Surprise**: 随学习降低
- **Step Count**: 总更新次数

---

## 常见问题

### Q1: L-UAV无法连接H-UAV

```bash
# 检查H-UAV是否运行
netstat -tlnp | grep 50051

# 测试连接
nc -zv localhost 50051

# 检查防火墙
sudo ufw allow 50051/tcp
```

### Q2: CUDA内存不足

```bash
# 使用8-bit量化
python hierarchical_uav/eval_task1.py --load_8bit

# 减小记忆维度
python hierarchical_uav/eval_task1.py --memory_dim 2048

# 减少token数量
python hierarchical_uav/eval_task1.py \
    --num_persistent_tokens 32 \
    --num_memory_tokens 64
```

### Q3: 调整通信阈值

```bash
# 更保守（更多远程查询）
python hierarchical_uav/eval_task1.py --threshold 0.5

# 更激进（更多本地决策）
python hierarchical_uav/eval_task1.py --threshold 0.9
```

---

## 监控和调试

### 实时查看H-UAV日志

```bash
tail -f outputs/hierarchical_uav/huav.log

# 输出示例：
# [H-UAV Stats] Requests: 10, Cache hit rate: 20.00%
# [H-UAV Stats] Requests: 50, Cache hit rate: 42.00%
```

### 实时查看L-UAV日志

```bash
tail -f outputs/hierarchical_uav/luav.log

# 输出示例：
# Self-match score: 0.65 → Query H-UAV
# Self-match score: 0.85 → Local decision
```

### Python监控

```python
# 获取H-UAV统计
from hierarchical_uav.models import LLaVAWithMAC

model = LLaVAWithMAC(config)
stats = model.get_memory_stats()

print(f"Cache size: {stats['cache_size']}")
print(f"Steps: {stats['step_count']}")
print(f"Surprise: {stats['surprise']:.4f}")
```

---

## 下一步

1. **查看完整文档**: `HIERARCHICAL_UAV_README.md`
2. **理解MAC原理**: `hierarchical_uav/mac_memory/`
3. **自定义配置**: 修改 `hierarchical_uav/models/uav_config.py`
4. **扩展功能**: 添加新的任务评估

---

## 快速命令参考

```bash
# 测试组件
python hierarchical_uav/mac_memory/neural_memory.py
python hierarchical_uav/communication/self_matching.py

# 启动H-UAV
./run_hierarchical_uav.sh h-uav

# 启动L-UAV
./run_hierarchical_uav.sh l-uav

# 自动运行
./run_hierarchical_uav.sh both

# 查看结果
cat outputs/hierarchical_uav/luav_results.jsonl | jq
```

---

**祝实验顺利！🚁✨**
