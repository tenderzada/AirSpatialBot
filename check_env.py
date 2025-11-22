#!/usr/bin/env python3
"""
Environment Check for Memory Injection
Verifies that all dependencies and paths are correctly configured.
"""

import sys
import os

def check_environment():
    """Check if environment is ready for memory injection."""

    print("=" * 60)
    print("Environment Check for Memory Injection")
    print("=" * 60)

    all_passed = True

    # 1. Check Python version
    print("\n[1/7] Checking Python version...")
    py_version = sys.version_info
    if py_version >= (3, 8):
        print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"✗ Python {py_version.major}.{py_version.minor} (need >= 3.8)")
        all_passed = False

    # 2. Check PyTorch
    print("\n[2/7] Checking PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"  ✓ CUDA available")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  ✓ GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
            if gpu_count < 2:
                print(f"  ⚠ Only {gpu_count} GPU available (recommend 2 for H-UAV + L-UAV)")
        else:
            print("  ✗ CUDA not available")
            all_passed = False
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install: pip install torch torchvision")
        all_passed = False

    # 3. Check Transformers
    print("\n[3/7] Checking Transformers...")
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        version_parts = transformers.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        if major < 4 or (major == 4 and minor < 37):
            print(f"  ⚠ Transformers {transformers.__version__} (recommend >= 4.37.0)")
    except ImportError:
        print("✗ Transformers not installed")
        print("  Install: pip install transformers>=4.37.0")
        all_passed = False

    # 4. Check LLaVA
    print("\n[4/7] Checking LLaVA...")
    try:
        from llava.model.builder import load_pretrained_model
        from llava.conversation import conv_templates
        print("✓ LLaVA installed")
    except ImportError as e:
        print("✗ LLaVA not installed")
        print("  Install: pip install git+https://github.com/haotian-liu/LLaVA.git")
        all_passed = False

    # 5. Check model path
    print("\n[5/7] Checking model paths...")
    model_path = "/mnt/data/AirSpatialBot"
    vision_tower = "/mnt/data/clip-vit-large-patch14-336"

    if os.path.exists(model_path):
        # Check for essential files
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            print(f"✓ Model found: {model_path}")
            print(f"  ✓ config.json exists")
        else:
            print(f"⚠ Model directory exists but config.json not found")
            print(f"  Path: {model_path}")
    else:
        print(f"✗ Model not found: {model_path}")
        print(f"  Please check your model path configuration")
        all_passed = False

    if os.path.exists(vision_tower):
        print(f"✓ Vision tower found: {vision_tower}")
    else:
        print(f"⚠ Vision tower not found: {vision_tower}")
        print("  Will try to download from HuggingFace if needed")

    # 6. Check test data
    print("\n[6/7] Checking test data...")
    test_data = "/mnt/data/AirSpatial/airspatial_agent_test_task1.jsonl"
    image_dir = "/mnt/data/AirSpatial/images"

    if os.path.exists(test_data):
        with open(test_data, 'r') as f:
            num_lines = sum(1 for _ in f)
        print(f"✓ Test data found: {test_data} ({num_lines} samples)")
    else:
        print(f"✗ Test data not found: {test_data}")
        all_passed = False

    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.JPG', '.png', '.PNG'))]
        num_images = len(image_files)
        print(f"✓ Image directory found: {image_dir} ({num_images} images)")
    else:
        print(f"✗ Image directory not found: {image_dir}")
        all_passed = False

    # 7. Check hierarchical_uav module
    print("\n[7/7] Checking hierarchical_uav module...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from hierarchical_uav.models import LLaVAWithMAC, UAVConfig
        from hierarchical_uav.communication import HUAVServer, LUAVClient
        print("✓ hierarchical_uav module accessible")
        print("  ✓ LLaVAWithMAC importable")
        print("  ✓ UAVConfig importable")
        print("  ✓ Communication modules importable")
    except ImportError as e:
        print(f"✗ hierarchical_uav import error: {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed!")
        print("=" * 60)
        print("\n🎉 You are ready to run memory injection!")
        print("\nNext steps:")
        print("  1. Make scripts executable:")
        print("     chmod +x run_huav.sh run_luav.sh")
        print("  2. Terminal 1 (H-UAV):")
        print("     ./run_huav.sh")
        print("  3. Terminal 2 (L-UAV):")
        print("     ./run_luav.sh")
    else:
        print("✗ Some checks failed")
        print("=" * 60)
        print("\n⚠ Please fix the issues above before running.")
        print("\nQuick fixes:")
        print("  • Install LLaVA: pip install git+https://github.com/haotian-liu/LLaVA.git")
        print("  • Install dependencies: pip install torch transformers accelerate bitsandbytes")
        print("  • Check paths in the error messages above")

    return all_passed


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
