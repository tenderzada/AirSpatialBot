# AirSpatialBot 完整设置指南

本指南提供了从零开始设置和运行 AirSpatialBot 的完整步骤。

## 📋 目录

1. [环境准备](#环境准备)
2. [LLaVA 集成](#llava-集成)
3. [数据准备](#数据准备)
4. [模型准备](#模型准备)
5. [运行评估](#运行评估)
6. [常见问题](#常见问题)

---

## 🔧 环境准备

### 系统要求

- **Python**: >= 3.8
- **CUDA**: >= 11.7 (推荐)
- **GPU**: >= 16GB VRAM (推荐)
- **磁盘空间**: >= 100GB

### 基础依赖安装

```bash
# 更新 pip
pip install --upgrade pip

# 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate pillow tqdm numpy scipy scikit-image opencv-python
```

---

## 📦 LLaVA 集成

### 自动安装（推荐）

```bash
# 一键安装 LLaVA
./install_llava.sh
```

### 手动安装

```bash
# 1. LLaVA 已经克隆在项目根目录
cd LLaVA

# 2. 安装 LLaVA
pip install -e .

# 3. 返回项目根目录
cd ..

# 4. 验证安装
python -c "from llava.model.builder import load_pretrained_model; print('✓ Success')"
```

详细文档：[LLAVA_INTEGRATION.md](LLAVA_INTEGRATION.md)

---

## 📊 数据准备

### 1. 下载数据集

从 HuggingFace 下载 AirSpatial 数据集：

```bash
# 安装 HuggingFace CLI
pip install huggingface_hub

# 下载数据集
huggingface-cli download erenzhou/AirSpatial --repo-type dataset --local-dir ./airspatial_data
```

### 2. 组织数据文件

将下载的文件移动到项目目录，然后运行：

```bash
./setup_data.sh
```

这会自动：
- 创建 `data/metadata/` 和 `data/images/` 目录
- 移动 JSONL 文件到正确位置
- 解压图像文件
- 统计数据

### 3. 数据文件说明

| 文件 | 用途 | 样本数 |
|------|------|--------|
| `airspatial_rec_train.jsonl` | Recognition 训练 | ~数千 |
| `airspatial_rec_test.jsonl` | Recognition 测试 | ~数百 |
| `airspatial_qa_train.jsonl` | VQA 训练 | ~数千 |
| `airspatial_sqa_test.jsonl` | Spatial QA 测试 | ~数百 |
| `airspatial_agent_test_task1.jsonl` | Agent 任务1 | ~数百 |
| `airspatial_agent_test_task2_v2.jsonl` | Agent 任务2 | ~数百 |
| `images.zip` | 航拍图像 | ~数千张 |

预期目录结构：

```
data/
├── metadata/
│   ├── airspatial_rec_test.jsonl
│   ├── airspatial_rec_train.jsonl
│   ├── airspatial_qa_train.jsonl
│   ├── airspatial_sqa_test.jsonl
│   ├── airspatial_agent_test_task1.jsonl
│   └── airspatial_agent_test_task2_v2.jsonl
└── images/
    └── airspatial/
        ├── *.jpg
        └── ...
```

---

## 🤖 模型准备

### 1. 下载 AirSpatialBot 模型

```bash
# 下载主模型
huggingface-cli download erenzhou/AirSpatialBot --local-dir ./models/AirSpatialBot
```

### 2. 准备本地 CLIP 模型

如果你已经有本地 CLIP 模型（如 `/mnt/data/clip-vit-large-patch14-336/`），跳过此步。

否则下载：

```bash
huggingface-cli download openai/clip-vit-large-patch14-336 \
    --local-dir /mnt/data/clip-vit-large-patch14-336
```

### 3. 测试 CLIP 模型

```bash
python test_local_clip.py
```

预期输出：
```
✅ 本地 CLIP 模型加载成功！
```

---

## 🚀 运行评估

### 快速开始

```bash
# 1. 确保所有依赖已安装
./install_llava.sh

# 2. 组织数据文件
./setup_data.sh

# 3. 测试 CLIP 模型
python test_local_clip.py

# 4. 运行评估
./run_eval_local_clip.sh
```

### 评估不同任务

#### 任务 1: 3D 空间定位 (Recognition)

```bash
# 使用默认配置
./run_eval_local_clip.sh
```

输出指标：
- 3D Precision @ 0.25
- BEV Precision @ 0.25
- Format error ratio

#### 任务 2: 空间问答 (SQA)

编辑 `run_eval_local_clip.sh`，修改：
```bash
TEST_FILE="./data/metadata/airspatial_sqa_test.jsonl"
OUTPUT_DIR="./outputs/eval_sqa"
```

然后运行：
```bash
./run_eval_local_clip.sh
```

#### 任务 3: Agent 任务

```bash
# 任务 1: 车辆属性识别
python main_task1.py \
    --test-file ./data/metadata/airspatial_agent_test_task1.jsonl \
    --image-folder ./data/images/ \
    --output-dir ./outputs/agent_task1/

# 任务 2: 车辆检索
python main_task2.py \
    --test-file ./data/metadata/airspatial_agent_test_task2_v2.jsonl \
    --image-folder ./data/images/ \
    --output-dir ./outputs/agent_task2/
```

---

## 📈 进阶使用

### 1. 生成训练数据

```bash
# 生成 VQA 训练数据
python llava_scripts/generate_sft/gen_sft_for_llava_3db_vqa.py

# 生成 Recognition 训练数据
python llava_scripts/generate_sft/gen_sft_for_llava_3db_rec.py
```

### 2. 微调模型

```bash
cd LLaVA

# 使用 LoRA 微调
bash scripts/v1_5/finetune_lora.sh

# 全参数微调
bash scripts/v1_5/finetune.sh
```

### 3. 部署模型服务

```bash
# 使用 vLLM
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model ./models/AirSpatialBot \
    --host 0.0.0.0 \
    --port 8000
```

---

## 🐛 常见问题

### 1. LLaVA 导入错误

**问题**：`ModuleNotFoundError: No module named 'llava'`

**解决**：
```bash
cd LLaVA
pip install -e .
cd ..
```

### 2. CLIP 模型下载

**问题**：脚本尝试从网络下载 CLIP

**解决**：
```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
./run_eval_local_clip.sh
```

### 3. CUDA 内存不足

**问题**：`CUDA out of memory`

**解决**：
```bash
# 方案 1: 使用 8-bit 量化
# 编辑评估脚本，添加 load_8bit=True

# 方案 2: 减小 batch size
# 在 run_eval_local_clip.sh 中设置 batch_size=1

# 方案 3: 使用 CPU（较慢）
CUDA_VISIBLE_DEVICES="" ./run_eval_local_clip.sh
```

### 4. 图像文件找不到

**问题**：`FileNotFoundError: [Errno 2] No such file or directory: 'xxx.jpg'`

**解决**：
```bash
# 确保图像已解压
ls data/images/airspatial/

# 重新组织数据
./setup_data.sh
```

### 5. transformers 版本冲突

**问题**：版本不兼容

**解决**：
```bash
pip install transformers==4.37.2 --force-reinstall
```

---

## 📁 完整项目结构

```
AirSpatialBot/
├── LLaVA/                              # LLaVA 仓库（通过 git clone 获得）
│   ├── llava/                          # LLaVA 核心模块
│   └── ...
├── data/
│   ├── metadata/                       # JSONL 标注文件
│   │   ├── airspatial_rec_test.jsonl
│   │   ├── airspatial_rec_train.jsonl
│   │   ├── airspatial_qa_train.jsonl
│   │   ├── airspatial_sqa_test.jsonl
│   │   ├── airspatial_agent_test_task1.jsonl
│   │   └── airspatial_agent_test_task2_v2.jsonl
│   └── images/                         # 图像文件
│       └── airspatial/
├── models/
│   └── AirSpatialBot/                  # AirSpatialBot 模型权重
├── outputs/                            # 评估结果
│   ├── eval_rec/
│   └── eval_sqa/
├── llava_scripts/                      # AirSpatialBot 的 LLaVA 脚本
│   ├── eval/
│   │   ├── batch_inference_3db.py
│   │   └── compute_metric_3db.py
│   └── generate_sft/
├── utils/                              # 工具函数
├── prompts/                            # 提示词模板
├── csv_file/                           # 车辆属性数据
├── main_task1.py                       # Agent 任务1
├── main_task2.py                       # Agent 任务2
├── llm_client.py                       # LLM 客户端
├── config.yml                          # 模型配置
├── install_llava.sh                    # LLaVA 安装脚本
├── setup_data.sh                       # 数据组织脚本
├── run_eval_local_clip.sh             # 评估运行脚本
├── test_local_clip.py                 # CLIP 测试脚本
├── .gitignore                         # Git 忽略文件
├── README.md                          # 项目说明
├── SETUP_GUIDE.md                     # 本文档
├── LLAVA_INTEGRATION.md               # LLaVA 集成文档
├── LOCAL_CLIP_SETUP.md                # 本地 CLIP 配置
└── QUICKSTART_LOCAL.md                # 快速入门
```

---

## ✅ 设置检查清单

完成设置后，确认：

- [ ] Python 环境已准备（>= 3.8）
- [ ] CUDA 可用（如有 GPU）
- [ ] LLaVA 已安装并可导入
- [ ] 数据文件已下载并组织
- [ ] AirSpatialBot 模型已下载
- [ ] 本地 CLIP 模型可用
- [ ] `test_local_clip.py` 运行成功
- [ ] 可以运行评估脚本

---

## 📚 相关文档

- **快速入门**: [QUICKSTART_LOCAL.md](QUICKSTART_LOCAL.md)
- **LLaVA 集成**: [LLAVA_INTEGRATION.md](LLAVA_INTEGRATION.md)
- **本地 CLIP 配置**: [LOCAL_CLIP_SETUP.md](LOCAL_CLIP_SETUP.md)
- **原始 README**: [README.md](README.md)

---

## 🎯 推荐的学习路径

### 初学者

1. 阅读 [QUICKSTART_LOCAL.md](QUICKSTART_LOCAL.md)
2. 运行 `./install_llava.sh`
3. 运行 `./setup_data.sh`
4. 运行 `./run_eval_local_clip.sh`
5. 查看评估结果

### 进阶用户

1. 理解 [LLAVA_INTEGRATION.md](LLAVA_INTEGRATION.md)
2. 生成训练数据
3. 微调 AirSpatialBot 模型
4. 部署模型服务
5. 开发自定义 Agent 任务

### 研究者

1. 阅读原始论文
2. 理解 3D 边界框和空间感知机制
3. 修改模型架构
4. 设计新的评估指标
5. 发布改进的模型

---

## 🤝 贡献

如果你对项目有改进建议或发现问题：

1. Fork 仓库
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

---

## 📄 许可证

本项目基于原始 AirSpatialBot 仓库，遵循其原始许可证。

LLaVA 遵循 Apache-2.0 许可证。

---

## 📧 联系方式

- **原始项目**: [AirSpatialBot GitHub](https://github.com/erenzhou/AirSpatialBot)
- **论文**: [IEEE TGRS](https://ieeexplore.ieee.org/document/11006099)
- **HuggingFace**: [erenzhou/AirSpatial](https://huggingface.co/datasets/erenzhou/AirSpatial)

---

🎉 **祝你使用愉快！**

如有任何问题，请参考相关文档或提交 Issue。
