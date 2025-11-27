"""
H-UAV Standalone SQA Evaluation with LoRA Memory Weaver

直接使用 LoRA 注入的记忆编织器进行推理，评估 MemGen 效果。

Architecture:
    图像 + 问题
         ↓
    ┌─────────────────────────────┐
    │  冻结 LLaVA (7B)             │
    └─────────────────────────────┘
         ↓
    ┌─────────────────────────────┐
    │  LoRA @ 层 8, 16, 24         │
    │  (q_proj, v_proj)            │
    └─────────────────────────────┘
         ↓
    ┌─────────────────────────────┐
    │  记忆增强的隐藏状态          │
    └─────────────────────────────┘
         ↓
    最终答案

Usage:
    python hierarchical_uav/eval_huav_standalone.py \
        --model_path /mnt/data/AirSpatialBot \
        --lora_weights ./outputs/lora_injection_sqa/lora_adapters_final.pt \
        --test_data /mnt/data/AirSpatial/airspatial_sqa_test.jsonl \
        --image_dir /mnt/data/AirSpatial/images \
        --output_dir ./outputs/eval_huav_standalone \
        --device cuda:0
"""

import argparse
import torch
import json
import os
import sys
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_uav.mac_memory.lora_injection import create_huav_memory_weaver
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN


def parse_numeric_answer(answer: str) -> Optional[float]:
    """Extract numeric value from answer."""
    # Remove common units
    answer = re.sub(r'(meters?|millimeters?|mm|m|cm)', '', answer, flags=re.IGNORECASE)

    # Find first number (int or float)
    match = re.search(r'-?\d+\.?\d*', answer)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def evaluate_huav_standalone(
    memory_weaver,
    tokenizer,
    image_processor,
    test_data: List[Dict],
    image_dir: str,
    output_path: str,
    device: str = 'cuda:0'
):
    """
    Evaluate H-UAV standalone on SQA task.

    Args:
        memory_weaver: LoRA-injected memory weaver
        tokenizer: Tokenizer
        image_processor: Image processor
        test_data: Test samples
        image_dir: Image directory
        output_path: Output path
        device: Device
    """
    print("=" * 70)
    print("H-UAV Standalone SQA Evaluation (LoRA Memory Weaver)")
    print("=" * 70)
    print(f"Total samples: {len(test_data)}")
    print(f"Device: {device}")
    print("=" * 70)
    print()

    results = []
    errors = []

    for i, sample in enumerate(tqdm(test_data, desc="H-UAV Standalone Evaluation")):
        try:
            question_id = sample.get('question_id', i)
            question = sample.get('question', '')
            qtype = sample.get('qtype', 'unknown')
            image_id = sample.get('image_id', '')
            ground_truth = sample.get('ground_truth', 0.0)

            # Load image
            image_path = os.path.join(image_dir, image_id)
            if not os.path.exists(image_path):
                if i < 10:  # Only log first 10 warnings
                    logger.warning(f"Image not found: {image_path}")
                continue

            image = Image.open(image_path).convert('RGB')

            # Prepare image tensor
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.to(device)

            # Ensure dtype consistency for 8-bit models
            try:
                # Try to get vision tower's dtype
                vision_tower = memory_weaver.base_model.get_model().get_vision_tower()
                if vision_tower is not None:
                    # Check if vision tower has a weight parameter to infer dtype
                    for param in vision_tower.parameters():
                        image_tensor = image_tensor.to(dtype=param.dtype)
                        break
            except Exception as e:
                # If dtype detection fails, use default float16 for 8-bit models
                logger.debug(f"Could not detect vision tower dtype: {e}")
                pass

            # Clean question
            question_clean = re.sub(r'<bbox>.*?</bbox>', '', question).strip()

            # Create type-specific prompt
            if qtype == 'depth':
                prompt = f"What is the depth of this vehicle in meters? Answer with just the numeric value."
            elif qtype == 'distance':
                prompt = f"How many meters is this vehicle from the camera? Answer with just the numeric value."
            elif qtype == 'length':
                prompt = f"What is the length of this vehicle in millimeters? Answer with just the numeric value."
            elif qtype == 'width':
                prompt = f"What is the width of this vehicle in millimeters? Answer with just the numeric value."
            elif qtype == 'height':
                prompt = f"What is the height of this vehicle in millimeters? Answer with just the numeric value."
            elif qtype == 'size':
                prompt = f"What are the length, width, and height in millimeters? Answer with three values."
            else:
                prompt = question_clean

            # Prepare conversation
            question_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], question_with_image)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()

            input_ids = tokenizer(prompt_text, return_tensors='pt')['input_ids'].to(device)

            # Generate with LoRA-injected model
            with torch.inference_mode():
                output_ids = memory_weaver.base_model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=512,
                    use_cache=True
                )

            # Decode
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # Extract answer
            if "ASSISTANT:" in outputs:
                answer = outputs.split("ASSISTANT:")[-1].strip()
            else:
                answer = outputs

            # Parse numeric answer
            predicted = parse_numeric_answer(answer)

            # Compute error
            if predicted is not None:
                error = abs(predicted - ground_truth)
                relative_error = error / max(abs(ground_truth), 1e-6)
            else:
                error = None
                relative_error = None

            results.append({
                'question_id': question_id,
                'qtype': qtype,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'answer_text': answer,
                'error': error,
                'relative_error': relative_error
            })

        except Exception as e:
            logger.error(f"Error on sample {i}: {e}")
            errors.append({'sample': i, 'error': str(e)})

    # Compute metrics
    valid_results = [r for r in results if r['predicted'] is not None]

    if valid_results:
        mae = np.mean([r['error'] for r in valid_results])
        rmse = np.sqrt(np.mean([r['error']**2 for r in valid_results]))
        mre = np.mean([r['relative_error'] for r in valid_results])

        # Compute per-type metrics
        type_metrics = {}
        for qtype in set(r['qtype'] for r in valid_results):
            type_results = [r for r in valid_results if r['qtype'] == qtype]
            if type_results:
                type_metrics[qtype] = {
                    'count': len(type_results),
                    'mae': float(np.mean([r['error'] for r in type_results])),
                    'rmse': float(np.sqrt(np.mean([r['error']**2 for r in type_results]))),
                    'mre': float(np.mean([r['relative_error'] for r in type_results]))
                }

        print()
        print("=" * 70)
        print("H-UAV Standalone Evaluation Results")
        print("=" * 70)
        print(f"Total samples: {len(test_data)}")
        print(f"Valid predictions: {len(valid_results)}")
        print(f"Failed predictions: {len(results) - len(valid_results)}")
        print()
        print("Overall Metrics:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MRE:  {mre:.4%}")
        print()
        print("Per-Type Metrics:")
        for qtype, metrics in sorted(type_metrics.items()):
            print(f"  {qtype:10s} (n={metrics['count']:4d}): MAE={metrics['mae']:8.2f}, RMSE={metrics['rmse']:8.2f}, MRE={metrics['mre']:7.2%}")
        print("=" * 70)

        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'metrics': {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'mre': float(mre)
                },
                'type_metrics': type_metrics,
                'results': results,
                'errors': errors
            }, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")

        return {
            'mae': mae,
            'rmse': rmse,
            'mre': mre,
            'valid_count': len(valid_results),
            'total_count': len(test_data)
        }

    else:
        print("\n❌ No valid predictions!")
        return None


def main():
    parser = argparse.ArgumentParser(description='H-UAV Standalone SQA Evaluation')

    # Model
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to base LLaVA model')
    parser.add_argument('--vision_tower', type=str, default=None,
                       help='Path to vision tower')
    parser.add_argument('--lora_weights', type=str, required=True,
                       help='Path to trained LoRA weights')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--load_8bit', action='store_true')

    # LoRA config
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--target_layers', type=int, nargs='+', default=[8, 16, 24])

    # Data
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)

    # Output
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    # Load test data
    logger.info(f"Loading test data from {args.test_data}...")
    with open(args.test_data) as f:
        test_data = [json.loads(line) for line in f]

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    logger.info(f"Loaded {len(test_data)} samples")

    # Create memory weaver with LoRA injection
    logger.info("Creating H-UAV memory weaver with LoRA injection...")
    memory_weaver = create_huav_memory_weaver(
        base_llava_path=args.model_path,
        device=args.device,
        lora_rank=args.lora_rank,
        target_layers=args.target_layers,
        load_8bit=args.load_8bit,
        vision_tower=args.vision_tower
    )

    # Load trained LoRA weights
    logger.info(f"Loading trained LoRA weights from {args.lora_weights}...")
    checkpoint = torch.load(args.lora_weights, map_location=args.device)

    # Load LoRA state
    memory_weaver.load_lora_state(checkpoint)

    # Get tokenizer and image processor from memory weaver
    tokenizer = memory_weaver.tokenizer
    image_processor = memory_weaver.image_processor

    logger.info("✓ Tokenizer and image processor ready")

    # Evaluate
    output_path = os.path.join(args.output_dir, 'results.json')
    evaluate_huav_standalone(
        memory_weaver=memory_weaver,
        tokenizer=tokenizer,
        image_processor=image_processor,
        test_data=test_data,
        image_dir=args.image_dir,
        output_path=output_path,
        device=args.device
    )


if __name__ == '__main__':
    main()
