"""
Quick test script to verify the memory injection fix.

This script loads the L-UAV model and tests generate_with_memory
with a small sample to ensure no shape errors occur.
"""

import torch
import sys
import os

# Add hierarchical_uav to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hierarchical_uav'))

from models.uav_config import UAVConfig
from models.llava_mac import LLaVAWithMAC

def test_memory_injection():
    """Test that generate_with_memory works without shape errors."""

    print("=" * 60)
    print("Testing Memory Injection Fix")
    print("=" * 60)

    # Create L-UAV config (adjust paths as needed)
    config = UAVConfig.create_luav_config(
        model_path="./models/AirSpatialBot",  # Update this path
        vision_tower="/mnt/data/clip-vit-large-patch14-336",  # Update this path
        device="cuda:0"
    )

    print(f"\n1. Loading L-UAV model...")
    print(f"   Device: {config.device}")
    print(f"   Model: {config.model_path}")

    try:
        model = LLaVAWithMAC(config)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False

    # Create dummy inputs
    print(f"\n2. Creating test inputs...")
    batch_size = 1
    seq_len = 51
    memory_dim = config.memory_dim

    # Dummy input_ids (with IMAGE_TOKEN_INDEX = -200 at position 10)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=config.device)
    input_ids[0, 10] = -200  # IMAGE_TOKEN placeholder

    # Dummy image (3-channel RGB)
    image_tensor = torch.randn(batch_size, 3, 224, 224, device=config.device)

    # Dummy memory features from H-UAV
    memory_features = torch.randn(batch_size, memory_dim, device=config.device)

    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Image shape: {image_tensor.shape}")
    print(f"   Memory features shape: {memory_features.shape}")
    print("   ✓ Test inputs created")

    # Test generate_with_memory
    print(f"\n3. Testing generate_with_memory...")
    try:
        with torch.no_grad():
            output_ids = model.generate_with_memory(
                input_ids=input_ids,
                images=image_tensor,
                memory_features=memory_features,
                memory_weight=0.5,
                max_new_tokens=10,  # Short generation for quick test
                do_sample=False
            )

        print(f"   ✓ Generation successful!")
        print(f"   Output shape: {output_ids.shape}")

    except RuntimeError as e:
        if "expected input" in str(e) and "channels" in str(e):
            print(f"   ✗ Shape error still present: {e}")
            return False
        else:
            print(f"   ✗ Runtime error: {e}")
            raise
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        raise

    print("\n" + "=" * 60)
    print("✓ All tests passed! Memory injection fix is working.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_memory_injection()
    sys.exit(0 if success else 1)
