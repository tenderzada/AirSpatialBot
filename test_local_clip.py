#!/usr/bin/env python3
"""
测试本地 CLIP 模型是否能正常加载
用于验证路径配置是否正确
"""

import os
import sys

# 设置离线模式，防止从网络下载
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

def test_clip_model(clip_path):
    """测试加载本地 CLIP 模型"""
    print("=" * 60)
    print("测试本地 CLIP 模型加载")
    print("=" * 60)
    print(f"CLIP 模型路径: {clip_path}")
    print()

    # 检查路径是否存在
    if not os.path.exists(clip_path):
        print(f"❌ 错误: 路径不存在: {clip_path}")
        return False

    print(f"✓ 路径存在")

    # 列出目录内容
    print(f"\n目录内容:")
    try:
        files = os.listdir(clip_path)
        for f in sorted(files)[:10]:  # 只显示前10个文件
            print(f"  - {f}")
        if len(files) > 10:
            print(f"  ... 还有 {len(files) - 10} 个文件")
    except Exception as e:
        print(f"❌ 无法读取目录: {e}")
        return False

    # 检查必要的文件
    print(f"\n检查必要文件:")
    required_files = [
        'config.json',
        'pytorch_model.bin',  # 或 model.safetensors
        'preprocessor_config.json'
    ]

    for req_file in required_files:
        if req_file in files or req_file.replace('.bin', '.safetensors') in files:
            print(f"  ✓ {req_file}")
        else:
            print(f"  ⚠ {req_file} (可能使用其他格式)")

    # 尝试加载模型
    print(f"\n尝试加载 CLIP 模型...")
    try:
        from transformers import CLIPImageProcessor, CLIPVisionModel

        print("  - 加载 Image Processor...")
        processor = CLIPImageProcessor.from_pretrained(clip_path, local_files_only=True)
        print(f"    ✓ Image Processor 加载成功")
        print(f"    - 图像尺寸: {processor.size}")

        print("  - 加载 Vision Model...")
        model = CLIPVisionModel.from_pretrained(clip_path, local_files_only=True)
        print(f"    ✓ Vision Model 加载成功")
        print(f"    - 隐藏层维度: {model.config.hidden_size}")
        print(f"    - 图像尺寸: {model.config.image_size}")

        print(f"\n{'=' * 60}")
        print("✅ 本地 CLIP 模型加载成功！")
        print("{'=' * 60}")
        return True

    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        print(f"\n可能的原因:")
        print(f"  1. 模型文件不完整")
        print(f"  2. 路径配置错误")
        print(f"  3. transformers 版本不兼容")
        return False

def main():
    # 默认 CLIP 模型路径
    clip_path = "/mnt/data/clip-vit-large-patch14-336/"

    # 如果提供了命令行参数，使用它
    if len(sys.argv) > 1:
        clip_path = sys.argv[1]

    success = test_clip_model(clip_path)

    if success:
        print(f"\n💡 使用建议:")
        print(f"   运行评估时添加参数: --vision-tower {clip_path}")
        print(f"   或使用提供的脚本: ./run_eval_local_clip.sh")
        sys.exit(0)
    else:
        print(f"\n💡 解决方案:")
        print(f"   1. 检查 CLIP 模型路径是否正确")
        print(f"   2. 确保模型文件完整下载")
        print(f"   3. 尝试重新下载 CLIP 模型")
        sys.exit(1)

if __name__ == "__main__":
    main()
