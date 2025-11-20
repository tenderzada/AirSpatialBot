# 🚀 AirSpatialBot 分层UAV系统 - 完整部署指南

从GitHub下载压缩包后的完整部署流程。

---

## 📦 第1步：解压和准备

```bash
# 1. 解压下载的zip文件
unzip AirSpatialBot-claude-review-forked-repo-*.zip

# 2. 进入项目目录
cd AirSpatialBot-*

# 3. 查看项目结构
ls -la
```

你应该看到：
```
.
├── hierarchical_uav/          # 分层UAV系统核心代码
├── llava_scripts/             # LLaVA评估脚本
├── LLaVA/                     # LLaVA仓库（需要重新克隆）
├── install_llava.sh           # LLaVA安装脚本
├── run_hierarchical_uav.sh    # 运行脚本
├── HIERARCHICAL_UAV_README.md # 完整文档
├── QUICKSTART_HIERARCHICAL.md # 快速开始
└── ...
```

---

## 🔧 第2步：环境准备

### 2.1 创建Python环境（推荐）

```bash
# 使用conda（推荐）
conda create -n airspatial python=3.10
conda activate airspatial

# 或使用venv
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 2.2 安装基础依赖

```bash
# 安装PyTorch (根据你的CUDA版本选择)
# CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# CPU only (不推荐，速度很慢)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# 安装其他基础依赖
pip install transformers==4.37.2 accelerate pillow tqdm numpy scipy scikit-image opencv-python
```

---

## 📥 第3步：克隆LLaVA（重要！）

由于GitHub不包含LLaVA子模块，需要手动克隆：

```bash
# 删除空的LLaVA目录（如果存在）
rm -rf LLaVA

# 重新克隆LLaVA
git clone https://github.com/haotian-liu/LLaVA.git

# 安装LLaVA
cd LLaVA
pip install -e .
cd ..

# 验证安装
python -c "from llava.model.builder import load_pretrained_model; print('✓ LLaVA installed')"
```

**或使用便捷脚本：**

```bash
# 运行安装脚本
chmod +x install_llava.sh
./install_llava.sh
```

---

## 💾 第4步：下载数据集

### 4.1 下载AirSpatial数据

```bash
# 安装HuggingFace CLI
pip install huggingface_hub

# 创建数据目录
mkdir -p data/metadata
mkdir -p data/images

# 下载数据集（约10-20GB）
huggingface-cli download erenzhou/AirSpatial \
    --repo-type dataset \
    --local-dir ./airspatial_data
```

### 4.2 组织数据文件

下载完成后，你会得到以下文件：
```
airspatial_data/
├── airspatial_agent_test_task1.jsonl
├── airspatial_agent_test_task2_v2.jsonl
├── airspatial_qa_train.jsonl
├── airspatial_rec_test.jsonl
├── airspatial_rec_train.jsonl
├── airspatial_sqa_test.jsonl
└── images.zip
```

运行数据组织脚本：

```bash
# 将数据文件移到当前目录
mv airspatial_data/*.jsonl ./
mv airspatial_data/images.zip ./

# 运行组织脚本
chmod +x setup_data.sh
./setup_data.sh
```

这会自动：
- 移动JSONL文件到 `data/metadata/`
- 解压images.zip到 `data/images/`
- 统计数据文件

---

## 🤖 第5步：下载模型

### 5.1 下载AirSpatialBot模型

```bash
# 创建模型目录
mkdir -p models

# 下载AirSpatialBot模型（约14GB）
huggingface-cli download erenzhou/AirSpatialBot \
    --local-dir ./models/AirSpatialBot
```

### 5.2 下载CLIP模型（本地使用）

```bash
# 下载CLIP vision tower
huggingface-cli download openai/clip-vit-large-patch14-336 \
    --local-dir ./models/clip-vit-large-patch14-336
```

如果你已经在其他地方有CLIP模型，可以创建符号链接：

```bash
ln -s /path/to/your/clip-vit-large-patch14-336 ./models/clip-vit-large-patch14-336
```

---

## ✅ 第6步：验证安装

运行快速测试：

```bash
# 测试LLaVA
python -c "from llava.model.builder import load_pretrained_model; print('✓ LLaVA OK')"

# 测试MAC模块
python -c "from hierarchical_uav.mac_memory import NeuralMemory; print('✓ MAC Memory OK')"

# 测试通信模块
python -c "from hierarchical_uav.communication import SelfMatchingGate; print('✓ Communication OK')"

# 测试模型配置
python -c "from hierarchical_uav.models import UAVConfig; print('✓ Models OK')"

# 测试CLIP模型
python test_local_clip.py
```

全部通过后显示：
```
✓ LLaVA OK
✓ MAC Memory OK
✓ Communication OK
✓ Models OK
✓ 本地 CLIP 模型加载成功！
```

---

## 🚀 第7步：运行系统

### 方法1：快速测试（小数据集）

```bash
# 创建小测试集（10个样本）
head -n 10 data/metadata/airspatial_agent_test_task1.jsonl > data/metadata/test_small.jsonl

# 运行测试（单GPU）
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --model_path ./models/AirSpatialBot \
    --vision_tower ./models/clip-vit-large-patch14-336 \
    --device cuda:0 \
    --test_data data/metadata/test_small.jsonl \
    --output ./outputs/test_results.jsonl
```

### 方法2：完整分层系统（2 GPU）

**Terminal 1 - 启动H-UAV（GPU 0）：**

```bash
# 赋予执行权限
chmod +x run_hierarchical_uav.sh

# 启动H-UAV服务器
./run_hierarchical_uav.sh h-uav
```

等待看到：
```
✓ H-UAV Server started
  Listening on port 50051
  Waiting for L-UAV connections...
```

**Terminal 2 - 启动L-UAV（GPU 1）：**

```bash
# 启动L-UAV客户端
./run_hierarchical_uav.sh l-uav
```

### 方法3：自动化运行

```bash
# 自动运行H-UAV和L-UAV
./run_hierarchical_uav.sh both
```

---

## 📝 第8步：修改配置（可选）

编辑 `run_hierarchical_uav.sh` 修改路径：

```bash
# 如果你的CLIP模型在其他位置
VISION_TOWER="./models/clip-vit-large-patch14-336"  # 改成你的路径

# 如果只有1个GPU
# 将L-UAV也放在GPU 0
# 在 eval_task1.py 中修改 --device cuda:0
```

---

## 🔍 第9步：查看结果

```bash
# 查看输出
ls -lh outputs/hierarchical_uav/

# 查看L-UAV结果
cat outputs/hierarchical_uav/luav_results.jsonl | head -5

# 使用jq美化输出（如果安装了jq）
cat outputs/hierarchical_uav/luav_results.jsonl | jq | head -20

# 查看日志
tail -f outputs/hierarchical_uav/luav.log
```

---

## 🐛 常见问题排查

### 问题1: ModuleNotFoundError: No module named 'llava'

**原因**: LLaVA未正确安装

**解决**:
```bash
cd LLaVA
pip install -e .
cd ..
```

### 问题2: FileNotFoundError: models/AirSpatialBot

**原因**: 模型未下载

**解决**:
```bash
huggingface-cli download erenzhou/AirSpatialBot --local-dir ./models/AirSpatialBot
```

### 问题3: CUDA out of memory

**解决**:
```bash
# 使用8-bit量化
python hierarchical_uav/eval_task1.py ... --load_8bit

# 或减小内存维度
python hierarchical_uav/eval_task1.py ... --memory_dim 2048
```

### 问题4: L-UAV无法连接H-UAV

**解决**:
```bash
# 检查H-UAV是否运行
netstat -tlnp | grep 50051

# 测试连接
nc -zv localhost 50051

# 或使用telnet
telnet localhost 50051
```

### 问题5: 缺少某些依赖

**解决**:
```bash
# 安装所有可能的依赖
pip install -r requirements.txt  # 如果有的话

# 或手动安装
pip install torch torchvision transformers accelerate pillow tqdm \
    numpy scipy scikit-image opencv-python huggingface_hub
```

---

## 📁 完整目录结构检查

运行前确保有以下结构：

```
AirSpatialBot/
├── hierarchical_uav/              ✓ 核心代码
│   ├── mac_memory/
│   ├── models/
│   ├── communication/
│   └── eval_task1.py
├── LLaVA/                         ✓ 需要克隆
│   └── llava/
├── data/                          ✓ 需要下载
│   ├── metadata/
│   │   └── *.jsonl
│   └── images/
│       └── airspatial/
├── models/                        ✓ 需要下载
│   ├── AirSpatialBot/
│   └── clip-vit-large-patch14-336/
├── outputs/                       ✓ 自动创建
├── install_llava.sh              ✓ 已有
├── run_hierarchical_uav.sh       ✓ 已有
└── HIERARCHICAL_UAV_README.md    ✓ 已有
```

---

## 🎯 最小化安装（快速开始）

如果你只想快速测试MAC模块而不运行完整系统：

```bash
# 1. 安装基础依赖
pip install torch torchvision transformers

# 2. 测试MAC记忆模块
python hierarchical_uav/mac_memory/neural_memory.py

# 3. 测试通信模块
python hierarchical_uav/communication/self_matching.py
```

---

## 📚 相关文档

- **完整文档**: `HIERARCHICAL_UAV_README.md`
- **快速开始**: `QUICKSTART_HIERARCHICAL.md`
- **LLaVA集成**: `LLAVA_INTEGRATION.md`
- **本地CLIP**: `LOCAL_CLIP_SETUP.md`

---

## 💡 下一步建议

1. **先小规模测试**（10个样本）
2. **验证H-UAV和L-UAV通信**
3. **运行完整评估**（500个样本）
4. **分析结果和统计数据**

---

## 🆘 需要帮助？

如果遇到问题：

1. 检查是否所有依赖都安装了
2. 确认模型和数据都下载了
3. 查看日志文件：`outputs/hierarchical_uav/*.log`
4. 参考完整文档中的故障排查部分

---

**祝你实验顺利！🚁✨**

如果有任何问题，随时查看文档或重新运行测试脚本。
