#!/usr/bin/env python3
"""
CLIP Vision Tower Configuration Diagnostic Tool

This script checks for mismatches between config.json, preprocessor_config.json,
and actual model weights (position embeddings).

Usage:
    python check_clip_config.py <path_to_clip_model>

Example:
    python check_clip_config.py /mnt/data/clip-vit-large-patch14-336
    python check_clip_config.py ./models/clip-vit-large-patch14-336
"""

import sys
import json
import os
from pathlib import Path
import torch


def check_config_json(model_path):
    """Check config.json"""
    config_path = os.path.join(model_path, 'config.json')

    print("=" * 80)
    print("1. Checking config.json")
    print("=" * 80)

    if not os.path.exists(config_path):
        print(f"❌ config.json not found at {config_path}")
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    image_size = config.get('image_size', 'NOT FOUND')
    patch_size = config.get('patch_size', 'NOT FOUND')

    print(f"Config path: {config_path}")
    print(f"Image size: {image_size}")
    print(f"Patch size: {patch_size}")

    if image_size != 'NOT FOUND' and patch_size != 'NOT FOUND':
        expected_patches = (image_size // patch_size) ** 2 + 1
        print(f"Expected position embeddings: {expected_patches}")
        print(f"  Calculation: ({image_size}/{patch_size})^2 + 1 = {expected_patches}")

    return config


def check_preprocessor_config(model_path):
    """Check preprocessor_config.json"""
    preprocessor_path = os.path.join(model_path, 'preprocessor_config.json')

    print("\n" + "=" * 80)
    print("2. Checking preprocessor_config.json")
    print("=" * 80)

    if not os.path.exists(preprocessor_path):
        print(f"❌ preprocessor_config.json not found at {preprocessor_path}")
        return None

    with open(preprocessor_path, 'r') as f:
        config = json.load(f)

    size = config.get('size', 'NOT FOUND')
    crop_size = config.get('crop_size', 'NOT FOUND')

    print(f"Preprocessor path: {preprocessor_path}")
    print(f"Size: {size}")
    print(f"Crop size: {crop_size}")

    return config


def check_model_weights(model_path):
    """Check actual position embeddings in model weights"""

    print("\n" + "=" * 80)
    print("3. Checking actual model weights (position embeddings)")
    print("=" * 80)

    # Try different weight file names
    weight_files = [
        'pytorch_model.bin',
        'model.safetensors',
        'pytorch_model.bin.index.json'
    ]

    weight_path = None
    for wf in weight_files:
        test_path = os.path.join(model_path, wf)
        if os.path.exists(test_path):
            weight_path = test_path
            break

    if weight_path is None:
        print(f"❌ No weight file found. Looked for: {weight_files}")
        return None

    print(f"Weight file: {weight_path}")

    # Load weights
    try:
        if weight_path.endswith('.bin'):
            weights = torch.load(weight_path, map_location='cpu')
            print(f"✓ Loaded weights from {weight_path}")
        elif weight_path.endswith('.safetensors'):
            from safetensors import safe_open
            weights = {}
            with safe_open(weight_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
            print(f"✓ Loaded weights from {weight_path}")
        else:
            print(f"❌ Unsupported weight file format: {weight_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return None

    # Find position embedding keys
    print("\nSearching for position embedding keys...")
    position_keys = [key for key in weights.keys() if 'position' in key.lower() and 'embed' in key.lower()]

    if not position_keys:
        print("❌ No position embedding keys found")
        print("\nAll keys in model:")
        for i, key in enumerate(weights.keys()):
            print(f"  {i+1}. {key}")
            if i >= 20:
                print(f"  ... and {len(weights.keys()) - 21} more keys")
                break
        return None

    print(f"✓ Found {len(position_keys)} position embedding key(s):")

    results = {}
    for key in position_keys:
        shape = weights[key].shape
        print(f"\n  Key: {key}")
        print(f"  Shape: {shape}")

        if len(shape) >= 2:
            num_positions = shape[0]
            embedding_dim = shape[1] if len(shape) > 1 else shape[0]

            print(f"  Number of positions: {num_positions}")
            print(f"  Embedding dimension: {embedding_dim}")

            # Infer resolution
            if num_positions == 257:
                resolution = 224
                patch_size = 14
                print(f"  → This corresponds to 224×224 resolution (patch_size=14)")
            elif num_positions == 577:
                resolution = 336
                patch_size = 14
                print(f"  → This corresponds to 336×336 resolution (patch_size=14)")
            elif num_positions == 197:
                resolution = 224
                patch_size = 16
                print(f"  → This corresponds to 224×224 resolution (patch_size=16)")
            else:
                # Try to infer
                sqrt_val = (num_positions - 1) ** 0.5
                if sqrt_val == int(sqrt_val):
                    patches_per_side = int(sqrt_val)
                    print(f"  → This corresponds to {patches_per_side}×{patches_per_side} patches")
                else:
                    print(f"  → Unknown resolution pattern")

            results[key] = {
                'shape': shape,
                'num_positions': num_positions,
                'embedding_dim': embedding_dim
            }

    return results


def check_with_transformers(model_path):
    """Try loading with transformers to see actual behavior"""

    print("\n" + "=" * 80)
    print("4. Testing with transformers library")
    print("=" * 80)

    try:
        from transformers import CLIPImageProcessor, CLIPVisionModel
        from PIL import Image
        import torch

        print("Loading image processor...")
        processor = CLIPImageProcessor.from_pretrained(model_path)
        print(f"✓ Processor loaded")
        print(f"  Size: {processor.size}")
        print(f"  Crop size: {processor.crop_size}")

        print("\nLoading vision model...")
        model = CLIPVisionModel.from_pretrained(model_path)
        print(f"✓ Model loaded")

        # Get actual position embedding shape from loaded model
        pos_emb = model.vision_model.embeddings.position_embedding.weight
        print(f"\nActual position embedding shape from loaded model:")
        print(f"  Shape: {pos_emb.shape}")
        print(f"  Number of positions: {pos_emb.shape[0]}")

        if pos_emb.shape[0] == 257:
            print(f"  → Model expects 224×224 images")
        elif pos_emb.shape[0] == 577:
            print(f"  → Model expects 336×336 images")

        # Test preprocessing
        print("\nTesting image preprocessing...")
        test_image = Image.new('RGB', (500, 500), color='red')
        processed = processor(images=test_image, return_tensors='pt')

        print(f"  Input image size: 500×500")
        print(f"  Processed tensor shape: {processed['pixel_values'].shape}")
        print(f"  → Images will be resized to: {processed['pixel_values'].shape[2]}×{processed['pixel_values'].shape[3]}")

        # Calculate expected patches
        img_size = processed['pixel_values'].shape[2]
        patch_size = 14  # Assuming patch size 14
        expected_patches = (img_size // patch_size) ** 2 + 1
        print(f"  Expected patches after encoding: {expected_patches}")

        return True

    except ImportError as e:
        print(f"❌ transformers library not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during transformers test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_clip_config.py <path_to_clip_model>")
        print("\nExample:")
        print("  python check_clip_config.py /mnt/data/clip-vit-large-patch14-336")
        print("  python check_clip_config.py ./models/clip-vit-large-patch14-336")
        sys.exit(1)

    model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"CLIP Vision Tower Configuration Check")
    print(f"Model path: {model_path}")
    print("=" * 80)

    # Run checks
    config = check_config_json(model_path)
    preprocessor = check_preprocessor_config(model_path)
    weights = check_model_weights(model_path)
    transformers_test = check_with_transformers(model_path)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    config_size = config.get('image_size', 'UNKNOWN') if config else 'UNKNOWN'
    preprocessor_size = preprocessor.get('size', 'UNKNOWN') if preprocessor else 'UNKNOWN'

    print(f"\nConfig.json image_size: {config_size}")
    print(f"Preprocessor size: {preprocessor_size}")

    if weights:
        for key, info in weights.items():
            num_pos = info['num_positions']
            print(f"Actual position embeddings: {num_pos}", end="")
            if num_pos == 257:
                print(" (224×224)")
            elif num_pos == 577:
                print(" (336×336)")
            else:
                print()

    # Detect mismatch
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if config and weights:
        config_size = config.get('image_size')

        # Get first weight result
        first_weight_key = list(weights.keys())[0]
        num_positions = weights[first_weight_key]['num_positions']

        expected_positions_224 = 257
        expected_positions_336 = 577

        if config_size == 224 and num_positions == 577:
            print("⚠️  MISMATCH DETECTED!")
            print("   config.json says 224×224")
            print("   but model weights have 577 position embeddings (336×336)")
            print("\n   → This will cause the error you're seeing!")
            print("   → The model was trained on 336×336 but config.json is wrong")
        elif config_size == 336 and num_positions == 257:
            print("⚠️  MISMATCH DETECTED!")
            print("   config.json says 336×336")
            print("   but model weights have 257 position embeddings (224×224)")
        elif config_size == 224 and num_positions == 257:
            print("✓ Config and weights are consistent (224×224)")
        elif config_size == 336 and num_positions == 577:
            print("✓ Config and weights are consistent (336×336)")
        else:
            print(f"⚠️  Unusual configuration:")
            print(f"   config.json: {config_size}")
            print(f"   weights: {num_positions} positions")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
