"""
H-UAV Baseline SQA Evaluation WITHOUT LoRA

直接使用基础 LLaVA 进行推理，作为消融实验的 baseline。

Architecture:
    图像 + 问题
         ↓
    ┌─────────────────────────────┐
    │  LLaVA (7B) - Baseline       │
    │  (No LoRA, No Memory)        │
    └─────────────────────────────┘
         ↓
    最终答案

Usage:
    python hierarchical_uav/eval_huav_baseline.py \
        --model_path /mnt/data/AirSpatialBot \
        --test_data /mnt/data/AirSpatial/airspatial_sqa_test.jsonl \
        --image_dir /mnt/data/AirSpatial/images \
        --output_dir ./outputs/ablation_huav_without_lora \
        --max_samples 100 \
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

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN


def parse_numeric_answer(answer: str) -> Optional[float]:
    """Extract numeric value from answer."""
    answer = re.sub(r'(meters?|millimeters?|mm|m|cm)', '', answer, flags=re.IGNORECASE)
    match = re.search(r'-?\d+\.?\d*', answer)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def parse_ground_truth(ground_truth, qtype: str) -> Optional[float]:
    """Parse ground truth value, handling different formats."""
    if isinstance(ground_truth, (int, float)):
        return float(ground_truth)

    gt_str = str(ground_truth)

    if qtype == 'size':
        match = re.search(r'<size>([\d.,\s]+)</size>', gt_str)
        if not match:
            match = re.search(r'size=([\d.,\s]+)', gt_str, re.IGNORECASE)
        if match:
            numbers = re.findall(r'[\d.]+', match.group(1))
            if numbers:
                return float(numbers[0])

    try:
        return float(gt_str)
    except ValueError:
        numbers = re.findall(r'[\d.]+', gt_str)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass

    return None


def infer_baseline(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    question: str,
    device: str = 'cuda:0'
) -> str:
    """Baseline inference without LoRA."""
    # Prepare image
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    image_tensor = image_tensor.to(device).to(torch.float16)

    # Prepare prompt
    question_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], question_with_image)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
            use_cache=True
        )

    # Decode
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    if "ASSISTANT:" in outputs:
        answer = outputs.split("ASSISTANT:")[-1].strip()
    else:
        answer = outputs

    return answer


def evaluate_baseline(
    model,
    tokenizer,
    image_processor,
    test_data: List[Dict],
    image_dir: str,
    output_path: str,
    device: str = 'cuda:0'
):
    """Evaluate baseline LLaVA (without LoRA)."""
    print("=" * 70)
    print("H-UAV Baseline Evaluation (WITHOUT LoRA)")
    print("=" * 70)
    print(f"Total samples: {len(test_data)}")
    print(f"Device: {device}")
    print("=" * 70)
    print()

    results = []
    errors = []

    for i, sample in enumerate(tqdm(test_data, desc="Baseline Evaluation")):
        try:
            question_id = sample.get('question_id', i)
            question = sample.get('question', '')
            qtype = sample.get('qtype', 'unknown')
            image_id = sample.get('image_id', '')

            # Parse ground truth
            ground_truth_raw = sample.get('ground_truth', 0.0)
            ground_truth = parse_ground_truth(ground_truth_raw, qtype)
            if ground_truth is None:
                logger.warning(f"Sample {i}: Could not parse ground_truth '{ground_truth_raw}', skipping")
                continue

            # Load image
            image_path = os.path.join(image_dir, image_id)
            if not os.path.exists(image_path):
                if i < 10:
                    logger.warning(f"Image not found: {image_path}")
                continue

            image = Image.open(image_path).convert('RGB')

            # Clean question
            question_clean = re.sub(r'<bbox>.*?</bbox>', '', question).strip()

            # Infer with baseline LLaVA
            answer = infer_baseline(model, tokenizer, image_processor, image, question_clean, device)

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
        print("Baseline Results (WITHOUT LoRA)")
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

        return {'mae': mae, 'rmse': rmse, 'mre': mre}

    else:
        print("\n❌ No valid predictions!")
        return None


def main():
    parser = argparse.ArgumentParser(description='H-UAV Baseline SQA Evaluation (WITHOUT LoRA)')

    # Model
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to base LLaVA model')
    parser.add_argument('--vision_tower', type=str, default=None,
                       help='Path to vision tower')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--load_8bit', action='store_true')

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

    # Load baseline LLaVA model
    logger.info("Loading baseline LLaVA model (WITHOUT LoRA)...")
    model_name = get_model_name_from_path(args.model_path)

    if 'cuda' in str(args.device):
        device_id = str(args.device).split(':')[-1] if ':' in str(args.device) else '0'
        device_map = {"": int(device_id)}
    else:
        device_map = "auto"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=args.load_8bit,
        device_map=device_map
    )

    if args.vision_tower and hasattr(model.config, 'mm_vision_tower'):
        model.config.mm_vision_tower = args.vision_tower

    # Ensure image_processor is loaded
    if image_processor is None:
        logger.warning("image_processor is None, loading manually...")
        from transformers import CLIPImageProcessor
        vision_path = args.vision_tower if args.vision_tower else args.model_path
        image_processor = CLIPImageProcessor.from_pretrained(vision_path)
        logger.info(f"✓ Loaded image_processor from {vision_path}")

    # Ensure vision tower is loaded
    vision_tower_obj = model.get_model().get_vision_tower()
    if hasattr(vision_tower_obj, 'load_model'):
        vision_tower_obj.load_model()

    logger.info(f"✓ Baseline LLaVA model loaded on {args.device}")

    # Evaluate
    output_path = os.path.join(args.output_dir, 'results.json')
    evaluate_baseline(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        test_data=test_data,
        image_dir=args.image_dir,
        output_path=output_path,
        device=args.device
    )


if __name__ == '__main__':
    main()
