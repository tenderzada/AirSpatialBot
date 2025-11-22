"""
Quick test to verify dtype fix for MAC layer with 8-bit models.

This script tests a single forward pass to ensure no dtype mismatches occur.
Run this before full training to verify the fix works.

Usage:
    python test_dtype_fix.py
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hierarchical_uav.models import LLaVAWithMAC, UAVConfig, UAVType
from PIL import Image
import numpy as np

def test_dtype_fix():
    """Test that MAC layer works with 8-bit models without dtype errors."""

    print("=" * 70)
    print("Testing dtype fix for MAC layer with 8-bit model")
    print("=" * 70)

    # Create H-UAV config with 8-bit loading (same as training)
    config = UAVConfig.create_huav_config(
        model_path="/mnt/data/AirSpatialBot",
        vision_tower="/mnt/data/clip-vit-large-patch14-336",
        device="cuda:0",
        memory_dim=4096,
        num_persistent_tokens=64,
        num_memory_tokens=128,
        load_8bit=True  # This triggers the dtype issue
    )

    print("\n1. Loading H-UAV model with 8-bit quantization...")

    # Clear CUDA cache to free up fragmented memory
    torch.cuda.empty_cache()
    print("   - Cleared CUDA cache")

    try:
        model = LLaVAWithMAC(config)

        # Enable gradient checkpointing to reduce memory usage
        # This trades compute for memory by recomputing activations during backward pass
        if hasattr(model.llava_model, 'enable_input_require_grads'):
            model.llava_model.enable_input_require_grads()
        if hasattr(model.llava_model.model, 'gradient_checkpointing_enable'):
            model.llava_model.model.gradient_checkpointing_enable()
            print("   - Enabled gradient checkpointing (saves memory)")

        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        return False

    # Check MAC layer dtype
    print("\n2. Checking MAC layer dtype...")
    if hasattr(model, 'mac_layer'):
        # Check a few key parameters
        ln_pre_weight_dtype = model.mac_layer.ln_pre.weight.dtype
        q_proj_weight_dtype = model.mac_layer.q_proj.weight.dtype
        memory_query_dtype = model.mac_layer.memory_query_proj.weight.dtype

        print(f"   - ln_pre.weight dtype: {ln_pre_weight_dtype}")
        print(f"   - q_proj.weight dtype: {q_proj_weight_dtype}")
        print(f"   - memory_query_proj.weight dtype: {memory_query_dtype}")

        if ln_pre_weight_dtype == torch.float32 and q_proj_weight_dtype == torch.float32:
            print("   ✓ MAC layer uses float32 (correct!)")
        else:
            print("   ✗ MAC layer not using float32 (will cause errors)")
            return False
    else:
        print("   ✗ MAC layer not found")
        return False

    # Create dummy input (proper way using image processor)
    print("\n3. Creating test input...")
    # Create a dummy PIL image instead of random tensor
    from PIL import Image as PILImage
    dummy_image = PILImage.new('RGB', (336, 336), color='red')

    # Process image properly
    image_tensor = model.image_processor.preprocess(dummy_image, return_tensors='pt')['pixel_values']
    image_tensor = image_tensor.to(config.device)
    print(f"   - Input image tensor shape: {image_tensor.shape}")

    # Check vision tower output dtype (should be float16 for 8-bit models)
    print("\n4. Checking vision tower output dtype...")
    with torch.no_grad():
        vision_features = model.llava_model.get_model().get_vision_tower()(image_tensor)
        vision_dtype = vision_features.dtype
        print(f"   - Vision features dtype: {vision_dtype}")
        if vision_dtype == torch.float16:
            print("   ✓ Vision tower outputs float16 (expected for 8-bit)")

        vision_features = model.llava_model.get_model().mm_projector(vision_features)
        projector_dtype = vision_features.dtype
        print(f"   - After mm_projector dtype: {projector_dtype}")

    # Test forward pass with MAC layer
    print("\n5. Testing forward pass through MAC layer...")

    # Clear cache before forward pass to maximize available memory
    torch.cuda.empty_cache()

    # Print memory stats
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"   - GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    try:
        # Test-time learning requires gradients, so NO torch.no_grad()!
        input_ids = model.tokenizer("Test question", return_tensors='pt')['input_ids'].to(config.device)
        outputs = model(
            input_ids=input_ids,
            images=image_tensor,  # Use properly processed image tensor
            update_memory=True  # Enable memory update like in training
        )
        print("   ✓ Forward pass successful - no dtype errors!")

        # Check memory metrics
        metrics = outputs.get('memory_metrics', {})
        print(f"\n6. Memory metrics:")
        print(f"   - Loss: {metrics.get('memory_loss', 0.0):.6f}")
        print(f"   - Surprise: {metrics.get('surprise', 0.0):.6f}")

        return True

    except RuntimeError as e:
        if "dtype" in str(e).lower():
            print(f"   ✗ Dtype mismatch error: {e}")
            print("\n   This means the fix didn't work properly.")
            print("   Please check:")
            print("     1. Did you pull the latest code?")
            print("     2. Are you using the correct Python environment?")
            print("     3. Try clearing cache: rm -rf ~/.cache/huggingface/")
        else:
            print(f"   ✗ Other error: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("\nQuick dtype fix verification")
    print("This will test one forward pass with 8-bit model + MAC layer\n")

    success = test_dtype_fix()

    print("\n" + "=" * 70)
    if success:
        print("✓ DTYPE FIX VERIFIED - Training should work now!")
        print("=" * 70)
        print("\nYou can now run:")
        print("  ./train_huav.sh")
    else:
        print("✗ DTYPE ISSUE STILL EXISTS - Training will fail")
        print("=" * 70)
        print("\nPlease report this output for further debugging.")

    sys.exit(0 if success else 1)
