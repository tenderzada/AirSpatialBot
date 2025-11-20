#!/bin/bash

# 优化的分层UAV运行脚本 - 解决GPU内存不足问题

# 配置
MODEL_PATH="./models/AirSpatialBot"
VISION_TOWER="./models/clip-vit-large-patch14-336"
TEST_DATA="./data/metadata/airspatial_agent_test_task1.jsonl"
OUTPUT_DIR="./outputs/hierarchical_uav"
PORT=50051

# 内存优化配置
USE_8BIT=true              # 使用8-bit量化
REDUCED_MEMORY=true        # 减小内存配置
MEMORY_DIM=2048           # 降低内存维度 (原4096)
NUM_PERSISTENT_TOKENS=32  # 降低token数 (原64)
NUM_MEMORY_TOKENS=64      # 降低token数 (原128)

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "优化的分层UAV系统 - 内存优化版"
echo "=========================================="
echo ""
echo "内存优化配置:"
echo "  - 8-bit量化: $USE_8BIT"
echo "  - 内存维度: $MEMORY_DIM (降低50%)"
echo "  - 持久token: $NUM_PERSISTENT_TOKENS (降低50%)"
echo "  - 内存token: $NUM_MEMORY_TOKENS (降低50%)"
echo ""

# 检查GPU
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# Function to start H-UAV
start_huav() {
    echo "=========================================="
    echo "启动 H-UAV (GPU 0) - 内存优化版"
    echo "=========================================="
    echo ""

    # 构建参数
    EXTRA_ARGS=""
    if [ "$USE_8BIT" = true ]; then
        EXTRA_ARGS="$EXTRA_ARGS --load_8bit"
    fi

    python hierarchical_uav/eval_task1.py \
        --uav_type h-uav \
        --model_path $MODEL_PATH \
        --vision_tower $VISION_TOWER \
        --device cuda:0 \
        --port $PORT \
        --test_data $TEST_DATA \
        --output $OUTPUT_DIR/huav_results.jsonl \
        --memory_dim $MEMORY_DIM \
        --num_persistent_tokens $NUM_PERSISTENT_TOKENS \
        --num_memory_tokens $NUM_MEMORY_TOKENS \
        $EXTRA_ARGS \
        2>&1 | tee $OUTPUT_DIR/huav_optimized.log
}

# Function to start L-UAV
start_luav() {
    echo "=========================================="
    echo "启动 L-UAV (GPU 1) - 内存优化版"
    echo "=========================================="
    echo ""

    # 等待H-UAV准备
    echo "等待H-UAV准备..."
    sleep 5

    # 检查H-UAV连接
    nc -zv localhost $PORT 2>&1 > /dev/null
    if [ $? -ne 0 ]; then
        echo "✗ 无法连接到H-UAV (localhost:$PORT)"
        echo "  请确保H-UAV已经启动"
        exit 1
    fi

    echo "✓ H-UAV已就绪"
    echo ""

    # 构建参数
    EXTRA_ARGS=""
    if [ "$USE_8BIT" = true ]; then
        EXTRA_ARGS="$EXTRA_ARGS --load_8bit"
    fi

    python hierarchical_uav/eval_task1.py \
        --uav_type l-uav \
        --model_path $MODEL_PATH \
        --vision_tower $VISION_TOWER \
        --device cuda:1 \
        --huav_address localhost:$PORT \
        --threshold 0.7 \
        --test_data $TEST_DATA \
        --output $OUTPUT_DIR/luav_results.jsonl \
        --memory_dim $MEMORY_DIM \
        --num_persistent_tokens $NUM_PERSISTENT_TOKENS \
        --num_memory_tokens $NUM_MEMORY_TOKENS \
        $EXTRA_ARGS \
        2>&1 | tee $OUTPUT_DIR/luav_optimized.log
}

# 主函数
case "$1" in
    h-uav)
        start_huav
        ;;
    l-uav)
        start_luav
        ;;
    both)
        echo "自动运行模式..."
        start_huav &
        HUAV_PID=$!
        sleep 10
        start_luav
        kill $HUAV_PID 2>/dev/null
        wait $HUAV_PID 2>/dev/null
        ;;
    *)
        echo "用法: $0 {h-uav|l-uav|both}"
        echo ""
        echo "内存优化版本 - 适用于GPU内存 < 24GB"
        echo ""
        echo "当前配置:"
        echo "  - 8-bit量化: $USE_8BIT"
        echo "  - 内存维度: $MEMORY_DIM"
        echo "  - Token数量: $NUM_PERSISTENT_TOKENS / $NUM_MEMORY_TOKENS"
        echo ""
        echo "如需进一步优化，编辑脚本修改配置"
        exit 1
        ;;
esac
