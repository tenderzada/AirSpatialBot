#!/bin/bash

# AirSpatialBot 数据组织脚本
# 用于将下载的数据文件组织到正确的目录结构

echo "=========================================="
echo "AirSpatialBot 数据组织脚本"
echo "=========================================="
echo ""

# 创建目录结构
echo "创建目录结构..."
mkdir -p data/metadata
mkdir -p data/images
mkdir -p models
mkdir -p outputs/eval_rec
mkdir -p outputs/eval_sqa
mkdir -p hf_cache

echo "✓ 目录创建完成"
echo ""

# 检查当前目录下的数据文件
echo "检查数据文件..."
echo ""

# JSONL 文件列表
JSONL_FILES=(
    "airspatial_agent_test_task1.jsonl"
    "airspatial_agent_test_task2_v2.jsonl"
    "airspatial_qa_train.jsonl"
    "airspatial_rec_test.jsonl"
    "airspatial_rec_train.jsonl"
    "airspatial_sqa_test.jsonl"
)

# 检查并移动 JSONL 文件
for file in "${JSONL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  找到: $file"
        if [ ! -f "data/metadata/$file" ]; then
            mv "$file" data/metadata/
            echo "    → 已移动到 data/metadata/"
        else
            echo "    → 已存在于 data/metadata/，跳过"
        fi
    elif [ -f "data/metadata/$file" ]; then
        echo "  ✓ $file (已在 data/metadata/)"
    else
        echo "  ✗ 未找到: $file"
    fi
done

echo ""

# 检查并解压图像
if [ -f "images.zip" ]; then
    echo "找到 images.zip，开始解压..."
    if [ ! -d "data/images/airspatial" ]; then
        unzip -q images.zip -d data/images/
        echo "✓ 图像解压完成"
    else
        echo "✓ 图像已解压，跳过"
    fi
elif [ -d "data/images/airspatial" ] || [ -d "data/images/images" ]; then
    echo "✓ 图像已存在"
else
    echo "⚠ 未找到 images.zip"
    echo "  请手动将图像放置到 data/images/ 目录"
fi

echo ""

# 统计数据
echo "=========================================="
echo "数据统计"
echo "=========================================="

# 统计 JSONL 文件行数
if [ -d "data/metadata" ]; then
    echo "JSONL 文件:"
    for file in data/metadata/*.jsonl; do
        if [ -f "$file" ]; then
            lines=$(wc -l < "$file")
            echo "  - $(basename $file): $lines 个样本"
        fi
    done
else
    echo "⚠ data/metadata 目录为空"
fi

echo ""

# 统计图像数量
if [ -d "data/images" ]; then
    image_count=$(find data/images -name "*.jpg" -o -name "*.png" | wc -l)
    echo "图像文件: $image_count 张"
else
    echo "⚠ data/images 目录为空"
fi

echo ""

# 检查模型
echo "模型检查:"
if [ -d "models/AirSpatialBot" ]; then
    echo "  ✓ AirSpatialBot 模型已存在"
else
    echo "  ⚠ AirSpatialBot 模型未找到"
    echo "    下载命令:"
    echo "    huggingface-cli download erenzhou/AirSpatialBot --local-dir ./models/AirSpatialBot"
fi

echo ""

# 检查本地 CLIP 模型
echo "CLIP 模型检查:"
CLIP_PATH="/mnt/data/clip-vit-large-patch14-336"
if [ -d "$CLIP_PATH" ]; then
    echo "  ✓ 本地 CLIP 模型已找到: $CLIP_PATH"
    # 运行测试脚本
    echo "  → 运行 CLIP 模型测试..."
    python test_local_clip.py "$CLIP_PATH"
else
    echo "  ⚠ 本地 CLIP 模型未找到: $CLIP_PATH"
    echo "    请确认 CLIP 模型路径，或下载:"
    echo "    huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir /mnt/data/clip-vit-large-patch14-336"
fi

echo ""
echo "=========================================="
echo "设置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 测试 CLIP 模型: python test_local_clip.py"
echo "  2. 运行评估: ./run_eval_local_clip.sh"
echo ""
echo "查看详细说明: cat LOCAL_CLIP_SETUP.md"
echo "=========================================="
