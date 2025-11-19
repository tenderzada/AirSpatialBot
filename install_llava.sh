#!/bin/bash

# LLaVA 集成和安装脚本
# 用于将 LLaVA 安装到 AirSpatialBot 项目中

echo "=========================================="
echo "LLaVA 集成和安装"
echo "=========================================="
echo ""

# 检查 LLaVA 目录是否存在
if [ ! -d "LLaVA" ]; then
    echo "⚠ LLaVA 目录不存在"
    echo "正在克隆 LLaVA 仓库..."
    git clone https://github.com/haotian-liu/LLaVA.git
    echo "✓ LLaVA 克隆完成"
else
    echo "✓ LLaVA 目录已存在"
fi

echo ""
echo "----------------------------------------"
echo "步骤 1/3: 安装 LLaVA 依赖"
echo "----------------------------------------"
echo ""

# 进入 LLaVA 目录并安装
cd LLaVA

# 安装为可编辑模式，这样 AirSpatialBot 可以导入它
echo "正在安装 LLaVA（可编辑模式）..."
pip install -e . --verbose

if [ $? -eq 0 ]; then
    echo "✓ LLaVA 核心安装完成"
else
    echo "✗ LLaVA 安装失败，请检查错误信息"
    exit 1
fi

echo ""
echo "----------------------------------------"
echo "步骤 2/3: 安装训练相关依赖（可选）"
echo "----------------------------------------"
echo ""

read -p "是否安装训练相关依赖（deepspeed, wandb）？[y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[train]"
    echo "✓ 训练依赖安装完成"
else
    echo "⊘ 跳过训练依赖安装"
fi

cd ..

echo ""
echo "----------------------------------------"
echo "步骤 3/3: 安装其他必要依赖"
echo "----------------------------------------"
echo ""

# 安装其他可能需要的依赖
pip install scikit-image opencv-python pillow tqdm shortuuid

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""

# 验证安装
echo "验证 LLaVA 安装..."
python -c "
try:
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token
    print('✓ LLaVA 模块导入成功！')
    print('  - constants')
    print('  - conversation')
    print('  - model.builder')
    print('  - mm_utils')
except ImportError as e:
    print(f'✗ LLaVA 导入失败: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "🎉 LLaVA 集成成功！"
    echo "=========================================="
    echo ""
    echo "现在你可以："
    echo "  1. 运行评估脚本: ./run_eval_local_clip.sh"
    echo "  2. 测试 LLaVA 模型加载"
    echo "  3. 开始使用 AirSpatialBot"
    echo ""
    echo "LLaVA 目录: $(pwd)/LLaVA"
    echo "=========================================="
else
    echo ""
    echo "⚠ 安装验证失败，请检查错误信息"
    exit 1
fi
