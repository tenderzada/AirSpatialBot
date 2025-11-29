# 3-Way 对比实验快速指南 (200 样本)

## 🎯 实验目标

对比三种方法在前 200 个样本上的性能：

| # | 实验 | 描述 | 文件 |
|---|------|------|------|
| 1 | **Baseline** | H-UAV 无 MemGen（vanilla LLaVA） | `eval_huav_baseline_200.sh` |
| 2 | **H-UAV + MemGen** | H-UAV 有 LoRA MemGen (10 epochs) | `eval_huav_standalone_200.sh` |
| 3 | **L-UAV + H-UAV** | L-UAV 与 H-UAV 协作 (每5样本query) | `eval_luav_huav_collab_200.sh` |

## 🚀 快速开始

### 一键运行（推荐）

```bash
# 确保 H-UAV server 正在运行
./start_huav_lora_socket.sh  # Terminal 1

# 运行完整 3-way 对比
./run_collaboration_comparison.sh  # Terminal 2
```

预计时间: **45-60 分钟**

### 分步运行

```bash
# 实验 1: Baseline (无 MemGen)
./eval_huav_baseline_200.sh

# 实验 2: H-UAV + MemGen
./eval_huav_standalone_200.sh

# 实验 3: L-UAV + H-UAV 协作
./eval_luav_huav_collab_200.sh

# 生成 3-way 对比报告
python compare_collaboration_results_3way.py
```

## 📊 预期输出

```
==================================================================================================
                        3-Way Collaboration Comparison (200 samples)
==================================================================================================

📊 Overall Metrics Comparison:
==================================================================================================
Metric     Baseline            H-UAV+MemGen        L-UAV+H-UAV         MemGen Gain         Collab Gain
--------------------------------------------------------------------------------------------------
MAE        XXX.XX              XXX.XX              XXX.XX              ±XX.X%              ±XX.X%
RMSE       XXX.XX              XXX.XX              XXX.XX              ±XX.X%              ±XX.X%
MRE        XX.XX%              XX.XX%              XX.XX%              ±XX.X%              ±XX.X%
==================================================================================================

💡 Detailed Analysis:
--------------------------------------------------------------------------------------------------

1️⃣  Baseline → H-UAV with MemGen:
   ✅ MemGen IMPROVES performance:
      - MAE reduced by XX.XX%
      - MRE reduced by XX.XX%
   → LoRA-enhanced memory generation is effective

2️⃣  Baseline → L-UAV + H-UAV Collaboration:
   ✅ Collaboration IMPROVES performance:
      - MAE reduced by XX.XX%
      - MRE reduced by XX.XX%
   → L-UAV benefits from H-UAV memory guidance

3️⃣  H-UAV with MemGen vs L-UAV + H-UAV:
   [自动分析哪个方法最优]

📈 Summary Table:
==================================================================================================
Approach                       MAE             MRE             vs Baseline         Rank
--------------------------------------------------------------------------------------------------
[最佳方法]                     XXX.XX          XX.XX%          +XX.X%              🥇
[第二名]                       XXX.XX          XX.XX%          +XX.X%              🥈
Baseline (no MemGen)           XXX.XX          XX.XX%          0.0%                Baseline
==================================================================================================
```

## 📁 输出文件

```
./outputs/
├── comparison_huav_baseline_200/       # 实验 1: Baseline
│   └── results.json
├── comparison_huav_standalone_200/     # 实验 2: H-UAV + MemGen
│   └── results.json
└── comparison_luav_huav_collab_200/    # 实验 3: L-UAV + H-UAV
    └── results.json
```

## 🔑 关键问题

### Q1: MemGen 是否有效？
**看**: Baseline → H-UAV+MemGen 的 gain
- 正值 = MemGen 有效，减少了误差
- 负值 = MemGen 无效或有害

### Q2: 协作是否有益？
**看**: Baseline → L-UAV+H-UAV 的 gain
- 正值 = 协作有益
- 负值 = 协作反而降低性能

### Q3: 最佳方法是什么？
**看**: Summary Table 中的 Rank
- 🥇 = 最佳方法
- 对比脚本会自动推荐

## 🛠️ 故障排除

### H-UAV server 未运行

```bash
❌ H-UAV server not running!

解决:
./start_huav_lora_socket.sh
```

### LoRA 权重未找到

```bash
❌ LoRA weights not found

解决:
# 检查权重文件是否存在
ls ./outputs/lora_injection_sqa/lora_adapters_*.pt

# 如果不存在，重新训练
./train_lora_injection_sqa.sh
```

### CUDA OOM

```bash
解决:
1. 确保只在一个 GPU 上运行评估
2. 使用 8-bit 量化（已默认）
3. 减少 batch size（如果适用）
```

## 📊 查看详细结果

```bash
# 查看单个实验结果
cat ./outputs/comparison_huav_baseline_200/results.json | jq .
cat ./outputs/comparison_huav_standalone_200/results.json | jq .
cat ./outputs/comparison_luav_huav_collab_200/results.json | jq .

# 重新生成对比报告
python compare_collaboration_results_3way.py

# 提取关键指标
jq -s '{
  baseline: .[0].metrics,
  memgen: .[1].metrics,
  collab: .[2].metrics
}' \
  ./outputs/comparison_huav_baseline_200/results.json \
  ./outputs/comparison_huav_standalone_200/results.json \
  ./outputs/comparison_luav_huav_collab_200/results.json
```

## 🎯 预期结果模式

### 模式 1: MemGen 显著有效

```
Baseline:       MAE=150.00
H-UAV+MemGen:   MAE=120.00  (Gain: +20.0%)  🥇
L-UAV+H-UAV:    MAE=125.00  (Gain: +16.7%)  🥈

结论: LoRA MemGen 显著提升性能
```

### 模式 2: 协作最优

```
Baseline:       MAE=150.00
H-UAV+MemGen:   MAE=135.00  (Gain: +10.0%)  🥈
L-UAV+H-UAV:    MAE=125.00  (Gain: +16.7%)  🥇

结论: 协作机制额外提供价值
```

### 模式 3: 性能相近

```
Baseline:       MAE=150.00
H-UAV+MemGen:   MAE=128.00  (Gain: +14.7%)  🥇
L-UAV+H-UAV:    MAE=129.00  (Gain: +14.0%)  🥈

结论: 两种方法都有效，性能相当
```

## 💡 下一步

### 如果 MemGen 有效
- ✅ 扩展到全部 17,526 样本
- ✅ 调整 LoRA 超参数（rank, alpha, layers）
- ✅ 尝试更多 training epochs

### 如果协作最优
- ✅ 优化 query 策略（频率、threshold）
- ✅ 改进 memory token 集成方式
- ✅ 探索更深度的协作机制

### 如果效果不明显
- ⚠️ 检查训练过程（loss curve）
- ⚠️ 验证 LoRA 权重加载正确
- ⚠️ 调整 memory generation 策略

## 📝 实验检查清单

在运行实验前：

- [ ] H-UAV server 正在运行 (`./start_huav_lora_socket.sh`)
- [ ] LoRA 权重文件存在 (`lora_adapters_epochbest.pt`)
- [ ] 测试数据可访问 (`/mnt/data/AirSpatial/...`)
- [ ] 有足够的磁盘空间（~500MB for results）
- [ ] GPU 可用（cuda:0 for H-UAV, cuda:1 for L-UAV）

运行实验后：

- [ ] 3 个 results.json 文件都已生成
- [ ] 对比报告成功生成
- [ ] 理解了关键结果和 ranking
- [ ] 保存了输出用于论文/分析

## 🔬 高级用法

### 调整强制 query 频率

编辑 `eval_luav_huav_collab_200.sh`:
```bash
FORCE_HUAV_EVERY_N=5  # 改为 3, 10, 20 等
```

### 调整 confidence threshold

编辑 `eval_luav_huav_collab_200.sh`:
```bash
CONFIDENCE_THRESHOLD=0.5  # 改为 0.3, 0.7 等
```

### 增加样本数

编辑所有 3 个评估脚本:
```bash
MAX_SAMPLES=200  # 改为 500, 1000 等
```

---

**准备就绪！** 运行 `./run_collaboration_comparison.sh` 开始 3-way 对比实验。
