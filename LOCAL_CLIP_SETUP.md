# 使用本地 CLIP 模型运行 AirSpatialBot 评估

本文档说明如何使用本地 CLIP 模型运行 AirSpatialBot 评估，避免从网络下载。

## 📝 修改内容

### 1. 修改了 `llava_scripts/eval/batch_inference_3db.py`
- ✅ 添加了 `--vision-tower` 参数，支持指定本地 CLIP 模型路径
- ✅ 添加了环境变量设置，防止自动下载模型

### 2. 创建了便捷运行脚本
- ✅ `run_eval_local_clip.sh` - 评估脚本（使用本地CLIP）
- ✅ `test_local_clip.py` - CLIP模型测试脚本

---

## 🚀 快速开始

### 方法 1：使用便捷脚本（推荐）

```bash
# 1. 首先测试 CLIP 模型是否可以正常加载
python test_local_clip.py

# 2. 如果测试通过，运行评估
./run_eval_local_clip.sh
```

### 方法 2：手动运行（自定义参数）

```bash
# 设置环境变量
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 运行评估
CUDA_VISIBLE_DEVICES=0 \
python llava_scripts/eval/batch_inference_3db.py \
    --model-path ./models/AirSpatialBot \
    --vision-tower /mnt/data/clip-vit-large-patch14-336 \
    --question-file ./data/metadata/airspatial_rec_test.jsonl \
    --image-folder ./data/images/ \
    --answers-file ./outputs/predictions.jsonl \
    --batch_size 1
```

### 方法 3：修改脚本中的默认路径

编辑 `run_eval_local_clip.sh`，修改以下变量：

```bash
MODEL_PATH="你的模型路径"
VISION_TOWER="你的CLIP模型路径"
IMAGE_FOLDER="你的图像目录"
TEST_FILE="你的测试数据"
```

---

## 📂 目录结构

确保你的数据组织如下：

```
AirSpatialBot/
├── data/
│   ├── metadata/
│   │   ├── airspatial_rec_test.jsonl       # Recognition 测试
│   │   ├── airspatial_sqa_test.jsonl       # Spatial QA 测试
│   │   ├── airspatial_qa_train.jsonl       # QA 训练
│   │   └── airspatial_rec_train.jsonl      # Recognition 训练
│   └── images/
│       └── airspatial/
│           ├── *.jpg
│           └── ...
├── models/
│   └── AirSpatialBot/                       # AirSpatialBot 模型
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
└── outputs/
    └── eval_rec/                            # 评估输出
```

本地 CLIP 模型路径：
```
/mnt/data/clip-vit-large-patch14-336/
├── config.json
├── preprocessor_config.json
├── pytorch_model.bin (或 model.safetensors)
└── ...
```

---

## 🔧 配置选项

### 评估脚本参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model-path` | AirSpatialBot 模型路径 | `./models/AirSpatialBot` |
| `--vision-tower` | **本地 CLIP 模型路径** | `/mnt/data/clip-vit-large-patch14-336` |
| `--image-folder` | 图像目录 | `./data/images/` |
| `--question-file` | 测试数据 JSONL | `./data/metadata/airspatial_rec_test.jsonl` |
| `--answers-file` | 输出结果路径 | `./outputs/predictions.jsonl` |
| `--batch_size` | 批次大小 | `1` |
| `--conv-mode` | 对话模式 | `llava_v1` |
| `--temperature` | 生成温度 | `0.2` |

### 环境变量

```bash
# 防止从 HuggingFace 下载模型
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 设置本地缓存（可选）
export HF_HOME="./hf_cache"
```

---

## 📊 不同评估任务

### 1. 评估 3D 空间定位（Recognition）

```bash
# 修改 run_eval_local_clip.sh 中的 TEST_FILE
TEST_FILE="./data/metadata/airspatial_rec_test.jsonl"

./run_eval_local_clip.sh
```

### 2. 评估空间问答（SQA）

```bash
# 使用 SQA 测试集
CUDA_VISIBLE_DEVICES=0 \
python llava_scripts/eval/batch_inference_3db.py \
    --model-path ./models/AirSpatialBot \
    --vision-tower /mnt/data/clip-vit-large-patch14-336 \
    --question-file ./data/metadata/airspatial_sqa_test.jsonl \
    --image-folder ./data/images/ \
    --answers-file ./outputs/predictions_sqa.jsonl \
    --batch_size 1
```

---

## ⚠️ 常见问题

### 1. CLIP 模型加载失败

**错误信息**：
```
OSError: Can't load tokenizer/config from /mnt/data/clip-vit-large-patch14-336
```

**解决方案**：
```bash
# 运行测试脚本检查
python test_local_clip.py

# 检查路径权限
ls -la /mnt/data/clip-vit-large-patch14-336/

# 检查必要文件是否存在
ls /mnt/data/clip-vit-large-patch14-336/config.json
```

### 2. 仍然尝试从网络下载

**解决方案**：
```bash
# 确保设置了环境变量
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 检查环境变量
echo $TRANSFORMERS_OFFLINE
```

### 3. GPU 内存不足

**解决方案**：
```bash
# 保持 batch_size=1
# 或使用 CPU（较慢）
CUDA_VISIBLE_DEVICES="" python llava_scripts/eval/batch_inference_3db.py ...
```

### 4. 模型路径不存在

**解决方案**：
```bash
# 检查模型是否下载
ls ./models/AirSpatialBot/

# 如果没有，从 HuggingFace 下载
huggingface-cli download erenzhou/AirSpatialBot --local-dir ./models/AirSpatialBot
```

---

## 📈 预期输出

### 批量推理成功后

会在输出目录生成：
```
outputs/eval_rec/
└── predictions_rec.jsonl      # 预测结果
```

每行格式：
```json
{
  "question_id": "xxx",
  "image_id": "xxx.jpg",
  "answer": "[x, y, z, l, w, h, r]",
  "bbox": [...],
  "bbox_3d": [...],
  ...
}
```

### 评估指标输出

```
ALL:
3D Precision @ 0.25: 0.XX
BEV Precision @ 0.25: 0.XX
Format error ratio: 0.XX
```

---

## 💡 使用建议

1. **首次运行**：先用 `test_local_clip.py` 验证 CLIP 模型
2. **小规模测试**：先用几个样本测试，确保流程正确
3. **完整评估**：使用完整测试集进行评估
4. **可视化**：在 `compute_metric_3db.py` 中添加 `--vis-dir` 查看结果

---

## 📚 参考资料

- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
- [AirSpatialBot Paper](https://ieeexplore.ieee.org/document/11006099)
- [HuggingFace Dataset](https://huggingface.co/datasets/erenzhou/AirSpatial)
- [HuggingFace Model](https://huggingface.co/erenzhou/AirSpatialBot)
