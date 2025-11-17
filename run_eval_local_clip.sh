#!/bin/bash

# AirSpatialBot 评估脚本 - 使用本地 CLIP 模型
# 作用：评估 3D 空间定位能力，从本地加载 CLIP 模型避免网络下载

# ========== 配置路径 ==========
MODEL_PATH="./models/AirSpatialBot"              # AirSpatialBot 模型路径
VISION_TOWER="/mnt/data/clip-vit-large-patch14-336"  # 本地 CLIP 模型路径
IMAGE_FOLDER="./data/images/"                     # 图像目录
TEST_FILE="./data/metadata/airspatial_rec_test.jsonl"  # 测试数据
OUTPUT_DIR="./outputs/eval_rec"                   # 输出目录
GPU_NUM=0                                         # GPU 编号

# ========== 设置环境变量 ==========
# 防止从 HuggingFace 下载模型
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# 设置本地缓存目录（可选）
export HF_HOME="./hf_cache"

# ========== 创建输出目录 ==========
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "AirSpatialBot 模型评估"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "CLIP模型: $VISION_TOWER"
echo "测试数据: $TEST_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# ========== 运行批量推理 ==========
echo "步骤 1/2: 运行批量推理..."
CUDA_VISIBLE_DEVICES=$GPU_NUM \
python llava_scripts/eval/batch_inference_3db.py \
    --model-path $MODEL_PATH \
    --vision-tower $VISION_TOWER \
    --question-file $TEST_FILE \
    --image-folder $IMAGE_FOLDER \
    --answers-file $OUTPUT_DIR/predictions_rec.jsonl \
    --batch_size 1 \
    --conv-mode llava_v1 \
    --temperature 0.2

# 检查推理是否成功
if [ $? -eq 0 ]; then
    echo "✓ 批量推理完成！"
else
    echo "✗ 批量推理失败，请检查错误信息"
    exit 1
fi

# ========== 计算评估指标 ==========
echo ""
echo "步骤 2/2: 计算评估指标..."
python llava_scripts/eval/compute_metric_3db.py \
    --answers-file $OUTPUT_DIR/predictions_rec.jsonl \
    --image-folder $IMAGE_FOLDER

# 可选：生成可视化结果（取消注释以启用）
# python llava_scripts/eval/compute_metric_3db.py \
#     --answers-file $OUTPUT_DIR/predictions_rec.jsonl \
#     --image-folder $IMAGE_FOLDER \
#     --vis-dir $OUTPUT_DIR/visualizations/

echo ""
echo "=========================================="
echo "评估完成！结果保存在: $OUTPUT_DIR"
echo "=========================================="
