# 🚀 AirSpatialBot 快速入门（使用本地 CLIP 模型）

## ✅ 已完成的修改

为了支持本地 CLIP 模型，已对代码进行以下修改：

1. ✅ **修改了评估脚本** `llava_scripts/eval/batch_inference_3db.py`
   - 添加了 `--vision-tower` 参数
   - 支持从本地加载 CLIP 模型
   - 自动设置离线模式环境变量

2. ✅ **创建了便捷脚本**
   - `setup_data.sh` - 数据组织脚本
   - `run_eval_local_clip.sh` - 评估运行脚本
   - `test_local_clip.py` - CLIP 模型测试脚本

---

## 📋 三步快速开始

### 第 1 步：组织数据文件

```bash
# 运行数据组织脚本
./setup_data.sh
```

这个脚本会：
- ✓ 创建必要的目录结构
- ✓ 移动 JSONL 文件到 `data/metadata/`
- ✓ 解压 `images.zip` 到 `data/images/`
- ✓ 统计数据文件数量
- ✓ 检查模型是否存在

### 第 2 步：测试 CLIP 模型

```bash
# 测试本地 CLIP 模型是否可以正常加载
python test_local_clip.py
```

如果看到 `✅ 本地 CLIP 模型加载成功！`，说明配置正确。

### 第 3 步：运行评估

```bash
# 运行完整评估流程
./run_eval_local_clip.sh
```

这个脚本会：
- ✓ 加载本地 CLIP 模型
- ✓ 运行批量推理
- ✓ 计算评估指标
- ✓ 输出结果到 `outputs/eval_rec/`

---

## 🎯 预期输出

### 成功的推理输出

```
==========================================
AirSpatialBot 模型评估
==========================================
模型路径: ./models/AirSpatialBot
CLIP模型: /mnt/data/clip-vit-large-patch14-336
测试数据: ./data/metadata/airspatial_rec_test.jsonl
输出目录: ./outputs/eval_rec
==========================================

步骤 1/2: 运行批量推理...
100%|██████████| 500/500 [10:23<00:00, 1.25s/it]
✓ 批量推理完成！

步骤 2/2: 计算评估指标...
ALL:
3D Precision @ 0.25: 0.XX
BEV Precision @ 0.25: 0.XX
Format error ratio: 0.XX

==========================================
评估完成！结果保存在: ./outputs/eval_rec
==========================================
```

### 输出文件

```
outputs/eval_rec/
└── predictions_rec.jsonl    # 模型预测结果
```

---

## 🔧 自定义配置

### 修改测试数据集

编辑 `run_eval_local_clip.sh`，修改 `TEST_FILE` 变量：

```bash
# 评估 Recognition 任务
TEST_FILE="./data/metadata/airspatial_rec_test.jsonl"

# 评估 Spatial QA 任务
TEST_FILE="./data/metadata/airspatial_sqa_test.jsonl"

# 评估 Agent 任务1
TEST_FILE="./data/metadata/airspatial_agent_test_task1.jsonl"
```

### 修改 CLIP 模型路径

如果你的 CLIP 模型在其他位置，修改 `run_eval_local_clip.sh`：

```bash
VISION_TOWER="你的CLIP模型路径"
```

或直接在命令行中指定：

```bash
python llava_scripts/eval/batch_inference_3db.py \
    --model-path ./models/AirSpatialBot \
    --vision-tower /your/custom/clip/path \
    --question-file ./data/metadata/airspatial_rec_test.jsonl \
    ...
```

### 启用可视化

编辑 `run_eval_local_clip.sh`，取消注释可视化部分：

```bash
# 取消下面几行的注释
python llava_scripts/eval/compute_metric_3db.py \
    --answers-file $OUTPUT_DIR/predictions_rec.jsonl \
    --image-folder $IMAGE_FOLDER \
    --vis-dir $OUTPUT_DIR/visualizations/
```

---

## 📂 完整目录结构

```
AirSpatialBot/
├── data/
│   ├── metadata/                           # JSONL 标注文件
│   │   ├── airspatial_rec_test.jsonl
│   │   ├── airspatial_sqa_test.jsonl
│   │   ├── airspatial_qa_train.jsonl
│   │   ├── airspatial_rec_train.jsonl
│   │   ├── airspatial_agent_test_task1.jsonl
│   │   └── airspatial_agent_test_task2_v2.jsonl
│   └── images/                             # 图像文件
│       └── airspatial/
│           ├── *.jpg
│           └── ...
├── models/
│   └── AirSpatialBot/                      # AirSpatialBot 模型权重
├── outputs/
│   ├── eval_rec/                           # Recognition 评估结果
│   └── eval_sqa/                           # SQA 评估结果
├── llava_scripts/
│   ├── eval/
│   │   ├── batch_inference_3db.py          # ✅ 已修改
│   │   └── compute_metric_3db.py
│   └── generate_sft/
├── setup_data.sh                           # ✅ 新增：数据组织脚本
├── run_eval_local_clip.sh                  # ✅ 新增：评估运行脚本
├── test_local_clip.py                      # ✅ 新增：CLIP测试脚本
├── QUICKSTART_LOCAL.md                     # ✅ 新增：快速入门（本文档）
└── LOCAL_CLIP_SETUP.md                     # ✅ 新增：详细配置说明
```

外部依赖：
```
/mnt/data/clip-vit-large-patch14-336/       # 本地 CLIP 模型
```

---

## 🐛 常见问题排查

### 问题 1: CLIP 模型加载失败

```bash
# 运行测试脚本诊断
python test_local_clip.py

# 检查路径
ls -la /mnt/data/clip-vit-large-patch14-336/

# 检查必要文件
ls /mnt/data/clip-vit-large-patch14-336/config.json
```

### 问题 2: 找不到测试数据

```bash
# 检查数据文件
ls data/metadata/

# 运行数据组织脚本
./setup_data.sh
```

### 问题 3: 模型未找到

```bash
# 下载 AirSpatialBot 模型
huggingface-cli download erenzhou/AirSpatialBot \
    --local-dir ./models/AirSpatialBot
```

### 问题 4: GPU 内存不足

```bash
# 在 run_eval_local_clip.sh 中保持 batch_size=1
batch_size 1

# 或使用 CPU（较慢）
CUDA_VISIBLE_DEVICES="" ./run_eval_local_clip.sh
```

---

## 📊 不同评估场景

### 场景 1: 评估 3D 空间定位

```bash
# 使用默认配置
./run_eval_local_clip.sh
```

### 场景 2: 评估空间问答（SQA）

```bash
# 编辑 run_eval_local_clip.sh，修改：
TEST_FILE="./data/metadata/airspatial_sqa_test.jsonl"
OUTPUT_DIR="./outputs/eval_sqa"

# 运行
./run_eval_local_clip.sh
```

### 场景 3: 小规模测试

```bash
# 只测试前 10 个样本
head -n 10 data/metadata/airspatial_rec_test.jsonl > data/metadata/test_small.jsonl

# 修改脚本使用小数据集
TEST_FILE="./data/metadata/test_small.jsonl"
```

---

## 📝 核心修改说明

### batch_inference_3db.py 的修改

**添加的参数：**
```python
parser.add_argument("--vision-tower", type=str, default=None,
                    help="Path to local vision tower (CLIP) model")
```

**修改的加载逻辑：**
```python
if args.vision_tower:
    os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 防止下载
    tokenizer, model, image_processor, context_len = load_pretrained_model(...)
    if hasattr(model.config, 'mm_vision_tower'):
        model.config.mm_vision_tower = args.vision_tower
```

---

## 💡 下一步建议

完成评估后，你可以：

1. **分析结果**
   ```bash
   # 查看预测结果
   head data/metadata/predictions_rec.jsonl

   # 统计指标
   grep "Precision" outputs/eval_rec/*.log
   ```

2. **可视化结果**（如果启用了 --vis-dir）
   ```bash
   ls outputs/eval_rec/visualizations/
   ```

3. **运行 Agent 任务**
   - 配置 `config.yml` 中的模型服务
   - 运行 `main_task1.py` 或 `main_task2.py`

4. **微调模型**
   - 使用 `gen_sft_for_llava_3db_vqa.py` 生成训练数据
   - 使用 LLaVA 训练脚本微调

---

## 📚 相关文档

- **详细配置说明**: `LOCAL_CLIP_SETUP.md`
- **原始 README**: `README.md`
- **LLaVA 文档**: https://github.com/haotian-liu/LLaVA
- **论文**: https://ieeexplore.ieee.org/document/11006099

---

## ✅ 检查清单

开始之前，确保：

- [ ] 已下载 AirSpatialBot 模型到 `./models/AirSpatialBot/`
- [ ] 本地 CLIP 模型位于 `/mnt/data/clip-vit-large-patch14-336/`
- [ ] 已下载并解压数据集（6个JSONL + images.zip）
- [ ] 已运行 `./setup_data.sh` 组织数据
- [ ] 已运行 `python test_local_clip.py` 测试 CLIP 模型
- [ ] GPU 可用（或准备使用 CPU）

全部完成后，运行：
```bash
./run_eval_local_clip.sh
```

🎉 开始评估吧！
