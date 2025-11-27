#!/usr/bin/env python3
"""
Debug script to test vision tower loading.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

def test_vision_tower():
    model_path = "/mnt/data/AirSpatialBot"
    vision_tower_path = "/mnt/data/clip-vit-large-patch14-336"
    device = "cuda:0"

    print("="* 70)
    print("Testing Vision Tower Loading")
    print("=" * 70)

    # Load model
    print(f"\n1. Loading base LLaVA from {model_path}...")
    model_name = get_model_name_from_path(model_path)
    print(f"   Model name: {model_name}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=True,
        device_map={"": 0}
    )

    print("   ✓ Model loaded")

    # Check image processor
    print(f"\n2. Checking image processor...")
    if image_processor is None:
        print("   ⚠️  image_processor is None")
        from transformers import CLIPImageProcessor
        print(f"   Loading manually from {vision_tower_path}...")
        image_processor = CLIPImageProcessor.from_pretrained(vision_tower_path)
        print("   ✓ Loaded manually")
    else:
        print(f"   ✓ image_processor loaded: {type(image_processor)}")

    # Check vision tower
    print(f"\n3. Checking vision tower...")
    try:
        vision_tower = model.get_model().get_vision_tower()
        print(f"   ✓ Vision tower retrieved: {type(vision_tower)}")

        # Check vision tower attributes
        print(f"\n4. Vision tower attributes:")
        attrs = dir(vision_tower)
        key_attrs = [a for a in attrs if not a.startswith('_')]
        print(f"   Key attributes: {', '.join(key_attrs[:10])}...")

        # Check for load_model
        if hasattr(vision_tower, 'load_model'):
            print(f"\n5. Loading vision tower model...")
            vision_tower.load_model()
            print(f"   ✓ Vision tower loaded")
        else:
            print(f"\n5. No load_model method found")

        # Try to get dtype
        print(f"\n6. Checking vision tower dtype...")
        if hasattr(vision_tower, 'dtype'):
            print(f"   dtype attribute: {vision_tower.dtype}")
        else:
            print(f"   No direct dtype attribute")
            print(f"   Checking parameters...")
            for i, param in enumerate(vision_tower.parameters()):
                print(f"   First parameter dtype: {param.dtype}")
                break
            else:
                print(f"   No parameters found")

        print("\n" + "=" * 70)
        print("✓ All checks passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == '__main__':
    success = test_vision_tower()
    sys.exit(0 if success else 1)
