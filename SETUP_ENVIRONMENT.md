# 环境准备指南 - Memory Injection 分支

## 概述

您有一个基于 LLaVA 微调的模型 (`/mnt/data/AirSpatialBot`)，需要安装 LLaVA 包来加载和运行它。

---

## 快速设置（推荐）

### 步骤 1: 安装 LLaVA

#### 方法 A: 从源码安装（推荐）

```bash
# 1. Clone LLaVA 仓库
cd /mnt/data  # 或其他临时目录
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

# 2. 安装依赖
pip install --upgrade pip
pip install -e .

# 3. 验证安装
python -c "from llava.model.builder import load_pretrained_model; print('✓ LLaVA installed successfully')"
```

#### 方法 B: 使用 pip（如果可用）

```bash
pip install git+https://github.com/haotian-liu/LLaVA.git
```

---

### 步骤 2: 安装其他依赖

```bash
cd /home/user/AirSpatialBot

# 核心依赖
pip install torch>=2.0.0 torchvision
pip install transformers>=4.37.0
pip install accelerate
pip install bitsandbytes  # 用于 8-bit/4-bit 量化

# 图像处理
pip install pillow

# 其他工具
pip install tqdm pandas
```

---

### 步骤 3: 验证您的模型和数据路径

根据您的配置：

```bash
# 检查模型路径
ls -lh /mnt/data/AirSpatialBot

# 应该包含：
# - config.json
# - pytorch_model.bin 或 model safetensors files
# - tokenizer files
# - (可选) adapter_config.json (如果是 LoRA 微调)

# 检查 Vision Tower
ls -lh /mnt/data/clip-vit-large-patch14-336

# 检查测试数据
ls -lh /mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl

# 检查图片目录
ls /mnt/data/AirSpatial/images | head -10
```

---

## 运行配置

### 创建启动脚本

#### H-UAV 启动脚本

创建 `run_huav.sh`:

```bash
#!/bin/bash

# H-UAV Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TEST_DATA="/mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/hierarchical_uav"
PORT=50051

# Create output directory
mkdir -p $OUTPUT_DIR

# Run H-UAV
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cuda:0 \
    --port $PORT \
    --model_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --test_data $TEST_DATA \
    --image_dir $IMAGE_DIR \
    --load_8bit \
    --output $OUTPUT_DIR/task1_h-uav_results.jsonl

echo "H-UAV completed. Memory saved to $OUTPUT_DIR/task1_h-uav_results_memory.pt"
```

#### L-UAV 启动脚本

创建 `run_luav.sh`:

```bash
#!/bin/bash

# L-UAV Configuration
MODEL_PATH="/mnt/data/AirSpatialBot"
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"
TEST_DATA="/mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl"
IMAGE_DIR="/mnt/data/AirSpatial/images"
OUTPUT_DIR="./outputs/hierarchical_uav"
HUAV_ADDRESS="localhost:50051"
THRESHOLD=0.7

# Create output directory
mkdir -p $OUTPUT_DIR

# Run L-UAV
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device cuda:1 \
    --huav_address $HUAV_ADDRESS \
    --threshold $THRESHOLD \
    --model_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --test_data $TEST_DATA \
    --image_dir $IMAGE_DIR \
    --load_8bit \
    --output $OUTPUT_DIR/task1_l-uav_results.jsonl

echo "L-UAV completed. Results saved to $OUTPUT_DIR/task1_l-uav_results.jsonl"
```

#### 使用启动脚本

```bash
# 添加执行权限
chmod +x run_huav.sh run_luav.sh

# 终端 1: 启动 H-UAV
./run_huav.sh

# 终端 2: 启动 L-UAV (等 H-UAV 完全启动后)
./run_luav.sh
```

---

## 关于您的微调模型

### 模型兼容性

您的模型 `/mnt/data/AirSpatialBot` 是基于 LLaVA 微调的，应该包含：

1. **基础 LLaVA 架构**：语言模型 + Vision Tower + mm_projector
2. **微调权重**：针对 AirSpatial 任务优化的参数

### 我们的实现如何使用它

```python
# hierarchical_uav/models/llava_mac.py 中的加载过程

# 1. 加载您的微调模型（保留所有微调权重）
self.llava_model = load_pretrained_model(
    model_path="/mnt/data/AirSpatialBot",  # 您的微调模型
    vision_tower="/mnt/data/clip-vit-large-patch14-336"
)

# 2. 添加 MAC 层（新增的记忆机制，不影响原有权重）
self.mac_layer = MACLayer(...)

# 3. 添加记忆投影层（新增，用于记忆注入）
self.memory_to_visual_proj = nn.Linear(...)
```

**重要**：
- ✅ 您的微调权重**完全保留**
- ✅ MAC 层和记忆投影层是**新增组件**
- ✅ 不会覆盖或修改您微调好的 LLaVA 权重

---

## 常见问题

### Q1: 是否需要下载原始 LLaVA 模型？

**不需要**。您已有微调模型 `/mnt/data/AirSpatialBot`，它包含了所有必要的权重。
只需要安装 LLaVA **代码库**（用于加载模型的函数）。

### Q2: 安装 LLaVA 会占用多少空间？

- LLaVA 代码库: ~100MB
- 依赖包: ~2-3GB (torch, transformers 等)
- **不包括**模型权重（您已经有了）

### Q3: 如果我的模型是 LoRA 微调的？

如果您的模型使用了 LoRA（`adapter_config.json` 存在）：

```bash
# 需要额外安装 peft
pip install peft

# 运行时会自动加载 base model + LoRA adapter
```

### Q4: 内存占用预估

使用您的配置（8-bit 量化）：

| 组件 | 显存占用 | 说明 |
|------|---------|------|
| LLaVA 模型 (8-bit) | ~12GB | 您的微调模型 |
| MAC 层 | ~500MB | 新增记忆机制 |
| 激活值 | ~2-3GB | 推理时临时占用 |
| **总计** | **~15GB** | 每个 UAV |

**建议**：
- H-UAV (cuda:0): 16GB+ 显存
- L-UAV (cuda:1): 16GB+ 显存

---

## 验证环境

运行以下脚本验证环境是否就绪：

```bash
# 创建验证脚本
cat > check_env.py << 'EOF'
#!/usr/bin/env python3
import sys

def check_environment():
    """Check if environment is ready for memory injection."""

    print("=" * 60)
    print("Environment Check for Memory Injection")
    print("=" * 60)

    # 1. Check Python version
    print("\n[1/7] Checking Python version...")
    py_version = sys.version_info
    if py_version >= (3, 8):
        print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"✗ Python {py_version.major}.{py_version.minor} (need >= 3.8)")
        return False

    # 2. Check PyTorch
    print("\n[2/7] Checking PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ GPU count: {torch.cuda.device_count()}")
        else:
            print("  ⚠ CUDA not available (CPU only)")
    except ImportError:
        print("✗ PyTorch not installed")
        return False

    # 3. Check Transformers
    print("\n[3/7] Checking Transformers...")
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
        return False

    # 4. Check LLaVA
    print("\n[4/7] Checking LLaVA...")
    try:
        from llava.model.builder import load_pretrained_model
        print("✓ LLaVA installed")
    except ImportError:
        print("✗ LLaVA not installed")
        print("  Install: pip install git+https://github.com/haotian-liu/LLaVA.git")
        return False

    # 5. Check model path
    print("\n[5/7] Checking model paths...")
    import os
    model_path = "/mnt/data/AirSpatialBot"
    vision_tower = "/mnt/data/clip-vit-large-patch14-336"

    if os.path.exists(model_path):
        print(f"✓ Model found: {model_path}")
    else:
        print(f"✗ Model not found: {model_path}")
        return False

    if os.path.exists(vision_tower):
        print(f"✓ Vision tower found: {vision_tower}")
    else:
        print(f"⚠ Vision tower not found: {vision_tower}")
        print("  Will try to download from HuggingFace")

    # 6. Check test data
    print("\n[6/7] Checking test data...")
    test_data = "/mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl"
    image_dir = "/mnt/data/AirSpatial/images"

    if os.path.exists(test_data):
        print(f"✓ Test data found: {test_data}")
    else:
        print(f"✗ Test data not found: {test_data}")
        return False

    if os.path.exists(image_dir):
        num_images = len([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.JPG', '.png'))])
        print(f"✓ Image directory found: {image_dir} ({num_images} images)")
    else:
        print(f"✗ Image directory not found: {image_dir}")
        return False

    # 7. Check hierarchical_uav module
    print("\n[7/7] Checking hierarchical_uav module...")
    try:
        from hierarchical_uav.models import LLaVAWithMAC, UAVConfig
        print("✓ hierarchical_uav module accessible")
    except ImportError as e:
        print(f"✗ hierarchical_uav import error: {e}")
        return False

    # Success
    print("\n" + "=" * 60)
    print("✓ All checks passed!")
    print("=" * 60)
    print("\nYou are ready to run memory injection!")
    print("\nNext steps:")
    print("  1. Terminal 1: ./run_huav.sh")
    print("  2. Terminal 2: ./run_luav.sh")

    return True

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
EOF

# 运行验证
python check_env.py
```

---

## 如果验证失败

### 安装缺失的包

```bash
# PyTorch (根据您的 CUDA 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Transformers
pip install transformers>=4.37.0

# LLaVA
pip install git+https://github.com/haotian-liu/LLaVA.git

# 其他依赖
pip install accelerate bitsandbytes pillow tqdm pandas
```

---

## 总结

**您需要做的**：

1. ✅ **安装 LLaVA 代码库**（~5 分钟）
   ```bash
   pip install git+https://github.com/haotian-liu/LLaVA.git
   ```

2. ✅ **安装依赖包**（~10 分钟）
   ```bash
   pip install torch transformers accelerate bitsandbytes
   ```

3. ✅ **验证环境**
   ```bash
   python check_env.py
   ```

4. ✅ **运行记忆注入功能**
   ```bash
   ./run_huav.sh  # 终端 1
   ./run_luav.sh  # 终端 2
   ```

**您不需要做的**：

❌ 下载原始 LLaVA 模型（您有微调版本）
❌ Clone 整个 LLaVA 项目用于训练（只需安装包）
❌ 修改您的微调模型（我们的代码在上层添加功能）

---

**准备好了吗？开始安装 LLaVA 吧！** 🚀
