#!/bin/bash

# GPU 内存诊断脚本

echo "=========================================="
echo "GPU 内存诊断"
echo "=========================================="
echo ""

# 检查NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "可用的GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader,nounits

    echo ""
    echo "当前GPU使用情况:"
    nvidia-smi

    echo ""
    echo "详细内存使用:"
    echo "GPU 0:"
    nvidia-smi -i 0 --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits
    echo "GPU 1:"
    nvidia-smi -i 1 --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits
else
    echo "未检测到NVIDIA GPU"
fi

echo ""
echo "=========================================="
echo "PyTorch GPU检测:"
python3 << EOF
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  总内存: {props.total_memory / 1024**3:.2f} GB")
        print(f"  已分配: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  已缓存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
EOF

echo ""
echo "=========================================="
echo "建议:"
echo "- 如果GPU内存 < 16GB: 使用8-bit量化"
echo "- 如果GPU内存 < 24GB: 减小MAC memory配置"
echo "- 如果两个GPU内存不同: 将大模型放在大内存GPU上"
echo "=========================================="
