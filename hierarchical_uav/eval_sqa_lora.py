"""
SQA Evaluation with LoRA Memory Weaver

L-UAV connects to H-UAV LoRA server to obtain memory tokens for enhanced inference.

Architecture:
    L-UAV (this):
      1. Base inference → evaluate confidence
      2. If low confidence → request memory from H-UAV
      3. Concatenate memory || input
      4. Enhanced inference → final answer

    H-UAV Server:
      - Receives image + question from L-UAV
      - Generates memory tokens via LoRA-injected LLaVA
      - Returns memory tokens [8, 4096]

Usage:
    # Standalone L-UAV (no H-UAV):
    python hierarchical_uav/eval_sqa_lora.py \
        --model_path /mnt/data/AirSpatialBot \
        --test_data /mnt/data/AirSpatial/airspatial_sqa_test.jsonl \
        --image_dir /mnt/data/AirSpatial/images \
        --output_dir ./outputs/eval_luav_standalone \
        --device cuda:1

    # L-UAV with H-UAV collaboration:
    python hierarchical_uav/eval_sqa_lora.py \
        --model_path /mnt/data/AirSpatialBot \
        --test_data /mnt/data/AirSpatial/airspatial_sqa_test.jsonl \
        --image_dir /mnt/data/AirSpatial/images \
        --output_dir ./outputs/eval_luav_with_huav \
        --device cuda:1 \
        --huav_url http://localhost:8000 \
        --confidence_threshold 0.5
"""

import argparse
import torch
import json
import os
import sys
import re
import requests
import base64
import io
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


class LUAVWithLoRAMemory:
    """
    L-UAV that can request memory from H-UAV LoRA server.
    """

    def __init__(
        self,
        model_path: str,
        vision_tower: Optional[str] = None,
        device: str = 'cuda:0',
        huav_url: Optional[str] = None,
        confidence_threshold: float = 0.5,
        load_8bit: bool = False
    ):
        self.device = device
        self.huav_url = huav_url
        self.confidence_threshold = confidence_threshold

        logger.info(f"Loading L-UAV model from {model_path}...")
        model_name = get_model_name_from_path(model_path)

        # Set device map
        if 'cuda' in str(device):
            device_id = str(device).split(':')[-1] if ':' in str(device) else '0'
            device_map = {"": int(device_id)}
        else:
            device_map = "auto"

        # Load model
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            load_8bit=load_8bit,
            device_map=device_map
        )

        # Override vision tower if specified
        if vision_tower and hasattr(self.model.config, 'mm_vision_tower'):
            self.model.config.mm_vision_tower = vision_tower

        logger.info(f"✓ L-UAV model loaded on {device}")

        if self.huav_url:
            logger.info(f"✓ Will use H-UAV memory from {self.huav_url}")
            logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        else:
            logger.info("⚠️  Running in standalone mode (no H-UAV)")

        self.stats = {
            'total_queries': 0,
            'huav_requests': 0,
            'standalone_inferences': 0
        }

    def request_memory_from_huav(
        self,
        image: Image.Image,
        question: str
    ) -> Optional[np.ndarray]:
        """
        Request memory tokens from H-UAV server.

        Args:
            image: PIL Image
            question: Question text

        Returns:
            Memory tokens [8, 4096] or None if request fails
        """
        if not self.huav_url:
            return None

        try:
            # Encode image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Send request
            response = requests.post(
                f"{self.huav_url}/get_memory",
                json={
                    'image': image_b64,
                    'question': question
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                memory_tokens = np.array(result['memory_tokens'])
                logger.debug(f"Received memory tokens: {memory_tokens.shape}")
                return memory_tokens
            else:
                logger.warning(f"H-UAV request failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error requesting memory from H-UAV: {e}")
            return None

    def infer(
        self,
        image: Image.Image,
        question: str,
        use_memory: bool = False,
        memory_tokens: Optional[np.ndarray] = None
    ) -> Tuple[str, float]:
        """
        Perform inference.

        Args:
            image: PIL Image
            question: Question text
            use_memory: Whether to use memory tokens
            memory_tokens: Memory tokens from H-UAV [8, 4096]

        Returns:
            (answer, confidence)
        """
        # Prepare image
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.to(self.device)

        # Prepare prompt
        question_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], question_with_image)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)

        # Generate
        with torch.inference_mode():
            # TODO: Integrate memory tokens if provided
            # For now, basic inference
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
                use_cache=True
            )

        # Decode
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Extract answer from response
        if "ASSISTANT:" in outputs:
            answer = outputs.split("ASSISTANT:")[-1].strip()
        else:
            answer = outputs

        # Simple confidence estimation (placeholder)
        # TODO: Implement proper confidence scoring
        confidence = 0.7 if not use_memory else 0.9

        return answer, confidence

    def evaluate_with_huav(
        self,
        image: Image.Image,
        question: str
    ) -> Tuple[str, float, bool]:
        """
        Evaluate with H-UAV collaboration.

        Flow:
            1. Base inference
            2. Check confidence
            3. If low confidence → request H-UAV memory
            4. Enhanced inference with memory

        Returns:
            (answer, confidence, used_huav)
        """
        self.stats['total_queries'] += 1

        # Step 1: Base inference
        answer_base, confidence_base = self.infer(image, question, use_memory=False)

        # Step 2: Check if H-UAV help is needed
        if self.huav_url and confidence_base < self.confidence_threshold:
            logger.debug(f"Low confidence ({confidence_base:.3f}), requesting H-UAV memory...")

            # Step 3: Request memory from H-UAV
            memory_tokens = self.request_memory_from_huav(image, question)

            if memory_tokens is not None:
                # Step 4: Enhanced inference with memory
                answer_enhanced, confidence_enhanced = self.infer(
                    image, question, use_memory=True, memory_tokens=memory_tokens
                )
                self.stats['huav_requests'] += 1
                return answer_enhanced, confidence_enhanced, True

        # Standalone inference
        self.stats['standalone_inferences'] += 1
        return answer_base, confidence_base, False


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


def evaluate_sqa_lora(
    luav: LUAVWithLoRAMemory,
    test_data: List[Dict],
    image_dir: str,
    output_path: str
):
    """
    Evaluate L-UAV on SQA task.

    Args:
        luav: L-UAV model
        test_data: Test samples
        image_dir: Image directory
        output_path: Output path
    """
    print("=" * 70)
    print("L-UAV SQA Evaluation with LoRA Memory Weaver")
    print("=" * 70)
    print(f"Total samples: {len(test_data)}")
    print(f"H-UAV URL: {luav.huav_url or 'None (standalone)'}")
    print(f"Confidence threshold: {luav.confidence_threshold}")
    print("=" * 70)
    print()

    results = []
    errors = []

    for i, sample in enumerate(tqdm(test_data, desc="L-UAV SQA Evaluation")):
        try:
            question_id = sample.get('question_id', i)
            question = sample.get('question', '')
            qtype = sample.get('qtype', 'unknown')
            image_id = sample.get('image_id', '')
            ground_truth = sample.get('ground_truth', 0.0)

            # Load image
            image_path = os.path.join(image_dir, image_id)
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue

            image = Image.open(image_path).convert('RGB')

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

            # Evaluate
            answer, confidence, used_huav = luav.evaluate_with_huav(image, prompt)

            # Parse answer
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
                'confidence': confidence,
                'used_huav': used_huav,
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

        print()
        print("=" * 70)
        print("Evaluation Results")
        print("=" * 70)
        print(f"Total samples: {len(test_data)}")
        print(f"Valid predictions: {len(valid_results)}")
        print(f"Failed predictions: {len(results) - len(valid_results)}")
        print()
        print("Metrics:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MRE:  {mre:.4%}")
        print()
        print("H-UAV Usage:")
        print(f"  Total queries: {luav.stats['total_queries']}")
        print(f"  H-UAV requests: {luav.stats['huav_requests']} ({luav.stats['huav_requests']/max(luav.stats['total_queries'],1)*100:.1f}%)")
        print(f"  Standalone: {luav.stats['standalone_inferences']} ({luav.stats['standalone_inferences']/max(luav.stats['total_queries'],1)*100:.1f}%)")
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
                'stats': luav.stats,
                'results': results,
                'errors': errors
            }, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='L-UAV SQA Evaluation with LoRA Memory')

    # Model
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vision_tower', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--load_8bit', action='store_true')

    # Data
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)

    # Output
    parser.add_argument('--output_dir', type=str, required=True)

    # H-UAV collaboration
    parser.add_argument('--huav_url', type=str, default=None,
                       help='H-UAV server URL (e.g., http://localhost:8000)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for requesting H-UAV help')

    args = parser.parse_args()

    # Load test data
    logger.info(f"Loading test data from {args.test_data}...")
    with open(args.test_data) as f:
        test_data = [json.loads(line) for line in f]

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    logger.info(f"Loaded {len(test_data)} samples")

    # Create L-UAV
    luav = LUAVWithLoRAMemory(
        model_path=args.model_path,
        vision_tower=args.vision_tower,
        device=args.device,
        huav_url=args.huav_url,
        confidence_threshold=args.confidence_threshold,
        load_8bit=args.load_8bit
    )

    # Evaluate
    output_path = os.path.join(args.output_dir, 'results.json')
    evaluate_sqa_lora(luav, test_data, args.image_dir, output_path)


if __name__ == '__main__':
    main()
