# 手动运行H-UAV和L-UAV评估指南

这个指南会帮你分步运行完整的评估流程。

## 准备工作

确保训练好的checkpoint存在：
```bash
ls -lh ./outputs/huav_training/huav_memory_final.pt
```

## 步骤 1: 启动H-UAV服务器（终端1）

在**第一个终端窗口**运行：

```bash
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cuda:0 \
    --port 50051 \
    --model_path /mnt/data/AirSpatialBot \
    --vision_tower /mnt/data/clip-vit-large-patch14-336 \
    --test_data /mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl \
    --image_dir /mnt/data/AirSpatial/images \
    --load_8bit \
    --load-memory ./outputs/huav_training/huav_memory_final.pt \
    --output ./outputs/hierarchical_uav/task1_h-uav_results.jsonl
```

**期望输出**：
```
============================================================
Starting H-UAV Evaluation
============================================================

Loading H-UAV model on cuda:0...
✓ H-UAV model loaded

Loading trained MAC memory from ./outputs/huav_training/huav_memory_final.pt...
✓ Trained memory loaded - H-UAV ready with learned representations

✓ H-UAV Server started
  Listening on port 50051
  Waiting for L-UAV connections...

Press Ctrl+C to stop server
```

**⏳ 等待**：H-UAV加载模型和checkpoint需要30-60秒。等到看到 "Listening on port 50051" 的消息。

**✅ 验证H-UAV已启动**：
在另一个终端运行：
```bash
nc -z localhost 50051 && echo "H-UAV is running" || echo "H-UAV not ready"
```

## 步骤 2: 运行L-UAV评估（终端2）

**等H-UAV完全启动后**，在**第二个终端窗口**运行：

```bash
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device cuda:1 \
    --huav_address localhost:50051 \
    --threshold 0.7 \
    --force-query-rate 0.3 \
    --model_path /mnt/data/AirSpatialBot \
    --vision_tower /mnt/data/clip-vit-large-patch14-336 \
    --test_data /mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl \
    --image_dir /mnt/data/AirSpatial/images \
    --load_8bit \
    --output ./outputs/hierarchical_uav/task1_l-uav_results.jsonl
```

**期望输出**：
```
============================================================
Starting L-UAV Evaluation
============================================================

Loading L-UAV model on cuda:1...
✓ L-UAV model loaded

Connecting to H-UAV at localhost:50051...
✓ Connected to H-UAV

Evaluating 971 samples...
[进度条显示...]
```

**说明**：
- `--force-query-rate 0.3` 强制30%的样本查询H-UAV（即使本地KB有答案）
- 评估大约需要10-20分钟（取决于GPU速度）

**监控终端1**：你应该能在H-UAV终端看到请求统计：
```
[H-UAV Stats] Requests: 10, Cache hit rate: 0.00%
[H-UAV Stats] Requests: 20, Cache hit rate: 5.00%
...
```

## 步骤 3: 停止H-UAV服务器

L-UAV评估完成后，回到**终端1**，按 `Ctrl+C` 停止H-UAV服务器。

**期望输出**：
```
^C
Shutting down H-UAV server...

============================================================
H-UAV Final Statistics
============================================================
Total requests served: 291
Cache hits: 15
Cache hit rate: 5.15%

✓ Memory state saved to ./outputs/hierarchical_uav/task1_h-uav_results_memory.pt
```

## 步骤 4: 分析结果

运行分析脚本：

```bash
python analyze_uav_results.py \
    --luav-results ./outputs/hierarchical_uav/task1_l-uav_results.jsonl \
    --ground-truth /mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl \
    --output-report ./outputs/hierarchical_uav/evaluation_report.json
```

**期望输出**：
```
Loading results...
  L-UAV: 971 results
  Ground truth: 971 samples

Calculating metrics...

================================================================================
UAV EVALUATION RESULTS - TRAINED MAC LAYERS
================================================================================

📊 OVERALL PERFORMANCE:
--------------------------------------------------------------------------------
Metric                         L-UAV
--------------------------------------------------------------------------------
Total Samples                  971
Exact Match Accuracy           XX.XX%
Partial Match Accuracy         XX.XX%
Unique Answers                 XXX

🧠 MEMORY USAGE ANALYSIS (L-UAV):
--------------------------------------------------------------------------------
  Total samples: 971
  Used H-UAV memory: ~291 (30.0%)
  Used local KB: ~680 (70.0%)
  Avg self-matching score: X.XXX

📈 ACCURACY BREAKDOWN:
...

🎯 KEY INSIGHTS:
--------------------------------------------------------------------------------
  Previous L-UAV accuracy (untrained MAC): ~3.1%
  Current L-UAV accuracy (trained MAC): XX.XX%
  Improvement: +XXX% (XXx better)

  Accuracy on memory-augmented samples: XX.XX%
  (Trained MAC helps L-UAV answer XXX/291 memory queries correctly)
================================================================================
```

## 预期结果

### ✅ 成功指标：
1. **Memory使用率**: ~30% (约291/971样本使用H-UAV)
2. **准确率提升**: 从3.1% → 期望30-50%
3. **Unique答案**: 200+ (vs 之前的8个)

### 📊 关键对比：

| 指标 | 训练前 | 训练后（期望） |
|------|--------|---------------|
| Memory使用率 | 33% | 30% |
| Unique答案 | 8个 | 200+ |
| 整体准确率 | 3.1% | 30-50% |
| Memory样本准确率 | ~0% | 20-40% |

## 常见问题

### Q1: H-UAV启动很慢
**A**: 正常。加载8-bit LLaVA + MAC checkpoint需要30-60秒。

### Q2: L-UAV报错 "Connection refused"
**A**: H-UAV还没完全启动。等到终端1显示 "Listening on port 50051"。

### Q3: Memory使用率仍然是0%
**A**: 检查L-UAV命令中是否包含 `--force-query-rate 0.3`。

### Q4: 准确率没有提升
**A**: 可能原因：
1. Checkpoint未正确加载
2. MAC层参数仍是随机的（检查训练日志）
3. Ground truth格式问题

## 下一步

如果评估结果满意，可以：
1. 调整 `--force-query-rate` 测试不同的memory使用比例
2. 调整 `--threshold` 改变自动触发memory查询的阈值
3. 分析哪些类型的问题受益于H-UAV memory
4. 创建PR总结实验结果
