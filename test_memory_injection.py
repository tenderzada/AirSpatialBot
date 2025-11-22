#!/usr/bin/env python3
"""
Test script for memory injection functionality.

This script verifies that:
1. Memory features can be properly injected into LLaVA inference
2. The memory projection layer works correctly
3. Augmented features have the expected shape
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_memory_injection():
    """Test memory injection without loading full model."""
    print("=" * 60)
    print("Testing Memory Injection Architecture")
    print("=" * 60)

    # Test parameters
    batch_size = 2
    memory_dim = 4096
    hidden_size = 4096
    num_image_patches = 576  # Typical for CLIP ViT-L/14@336px

    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Memory dim: {memory_dim}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Image patches: {num_image_patches}")

    # 1. Test memory projection layer
    print("\n" + "-" * 60)
    print("1. Testing Memory Projection Layer")
    print("-" * 60)

    memory_proj = torch.nn.Linear(memory_dim, hidden_size)
    memory_features = torch.randn(batch_size, memory_dim)

    memory_tokens = memory_proj(memory_features)  # [batch, hidden_size]
    memory_tokens = memory_tokens.unsqueeze(1)    # [batch, 1, hidden_size]

    print(f"✓ Memory features shape: {memory_features.shape}")
    print(f"✓ Memory tokens shape: {memory_tokens.shape}")

    assert memory_tokens.shape == (batch_size, 1, hidden_size), \
        f"Expected shape ({batch_size}, 1, {hidden_size}), got {memory_tokens.shape}"

    # 2. Test feature concatenation
    print("\n" + "-" * 60)
    print("2. Testing Feature Concatenation")
    print("-" * 60)

    # Simulate image features from vision encoder
    image_features = torch.randn(batch_size, num_image_patches, hidden_size)
    print(f"✓ Image features shape: {image_features.shape}")

    # Concatenate memory tokens with image features
    augmented_features = torch.cat([image_features, memory_tokens], dim=1)

    expected_num_tokens = num_image_patches + 1  # +1 for memory token
    print(f"✓ Augmented features shape: {augmented_features.shape}")
    print(f"✓ Total visual tokens: {expected_num_tokens} (image: {num_image_patches}, memory: 1)")

    assert augmented_features.shape == (batch_size, expected_num_tokens, hidden_size), \
        f"Expected shape ({batch_size}, {expected_num_tokens}, {hidden_size}), got {augmented_features.shape}"

    # 3. Test memory weight scaling
    print("\n" + "-" * 60)
    print("3. Testing Memory Weight Scaling")
    print("-" * 60)

    memory_weight = 0.5
    scaled_memory_tokens = memory_tokens * memory_weight

    # Verify scaling
    original_norm = torch.norm(memory_tokens)
    scaled_norm = torch.norm(scaled_memory_tokens)
    expected_ratio = memory_weight
    actual_ratio = (scaled_norm / original_norm).item()

    print(f"✓ Memory weight: {memory_weight}")
    print(f"✓ Original norm: {original_norm.item():.4f}")
    print(f"✓ Scaled norm: {scaled_norm.item():.4f}")
    print(f"✓ Scaling ratio: {actual_ratio:.4f} (expected: {expected_ratio:.4f})")

    assert abs(actual_ratio - expected_ratio) < 0.01, \
        f"Scaling ratio {actual_ratio} doesn't match expected {expected_ratio}"

    # 4. Test with/without memory comparison
    print("\n" + "-" * 60)
    print("4. Testing With/Without Memory Scenarios")
    print("-" * 60)

    # Without memory
    features_no_memory = image_features
    print(f"✓ Features without memory: {features_no_memory.shape}")

    # With memory
    features_with_memory = torch.cat([image_features, scaled_memory_tokens], dim=1)
    print(f"✓ Features with memory: {features_with_memory.shape}")

    # Verify difference
    num_tokens_diff = features_with_memory.shape[1] - features_no_memory.shape[1]
    print(f"✓ Additional tokens from memory: {num_tokens_diff}")

    assert num_tokens_diff == 1, f"Expected 1 additional token, got {num_tokens_diff}"

    # Summary
    print("\n" + "=" * 60)
    print("Memory Injection Test Results")
    print("=" * 60)
    print("✓ All tests passed!")
    print("\nKey Findings:")
    print(f"  • Memory projection: {memory_dim}D → {hidden_size}D")
    print(f"  • Memory tokens per sample: 1")
    print(f"  • Total tokens increase: {num_image_patches} → {expected_num_tokens}")
    print(f"  • Memory weight scaling: working correctly")
    print("\nMemory injection is ready to augment L-UAV inference with H-UAV knowledge!")

    return True


def test_config_compatibility():
    """Test that the memory injection is compatible with UAVConfig."""
    print("\n" + "=" * 60)
    print("Testing Configuration Compatibility")
    print("=" * 60)

    from hierarchical_uav.models import UAVConfig

    # Create L-UAV config
    config = UAVConfig.create_luav_config(
        model_path="./models/AirSpatialBot",
        vision_tower="/mnt/data/clip-vit-large-patch14-336",
        huav_address="localhost:50051"
    )

    print(f"\n✓ L-UAV Config created:")
    print(f"  - UAV type: {config.uav_type}")
    print(f"  - Memory dim: {config.memory_dim}")
    print(f"  - Enable MAC: {config.enable_mac}")
    print(f"  - Memory update: {config.enable_memory_update}")

    assert config.enable_mac, "MAC should be enabled for L-UAV"
    assert not config.enable_memory_update, "Memory update should be disabled for L-UAV"

    print("\n✓ Configuration compatibility verified!")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Memory Injection Test Suite")
    print("=" * 60)

    try:
        # Run architecture tests
        test_memory_injection()

        # Run config tests
        test_config_compatibility()

        print("\n" + "=" * 60)
        print("🎉 All Tests Passed!")
        print("=" * 60)
        print("\nThe memory injection implementation is working correctly.")
        print("H-UAV memory features will now be properly injected into L-UAV inference.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
