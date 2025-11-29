"""
SQA Evaluation with LoRA-based Hierarchical UAV Collaboration V3

L-UAV (3-epoch LoRA) requests answers from H-UAV (10-epoch LoRA) when needed

Architecture:
- L-UAV: 3-epoch LoRA for basic capability
- H-UAV: 10-epoch LoRA for enhanced capability
- L-UAV requests H-UAV's answer directly (not memory tokens)
- Simpler and more effective collaboration
"""

import argparse
import torch
import json
import os
import sys
import re
import socket
import pickle
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


class LUAVSocketClient:
    """L-UAV Socket Client for H-UAV Answer Server"""

    def __init__(
        self,
        huav_address: str = "localhost:50052",  # Different port for V3
        timeout: float = 30.0
    ):
        # Parse address
        if ':' in huav_address:
            self.host, port_str = huav_address.split(':')
            self.port = int(port_str)
        else:
            self.host = huav_address
            self.port = 50052

        self.timeout = timeout
        self.request_counter = 0

        logger.info(f"✓ L-UAV Socket Client initialized for H-UAV at {self.host}:{self.port}")

    def request_answer(
        self,
        image: Image.Image,
        question: str
    ) -> Optional[str]:
        """
        Request answer from H-UAV (10-epoch LoRA)

        Returns:
            Answer string or None
        """
        self.request_counter += 1

        try:
            # Create socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(self.timeout)

            # Connect to H-UAV
            client_socket.connect((self.host, self.port))

            # Prepare request
            import io
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            image_bytes = buf.getvalue()

            request = {
                'image': image_bytes,
                'question': question,
                'request_id': self.request_counter
            }

            # Serialize and send
            request_data = pickle.dumps(request)
            client_socket.sendall(request_data + b'<END>')

            logger.debug(f"Sent answer request #{self.request_counter} to H-UAV")

            # Receive response
            data = b''
            while True:
                chunk = client_socket.recv(4096)
                data += chunk

                if b'<END>' in data:
                    data = data.replace(b'<END>', b'')
                    break

                if not chunk and len(data) == 0:
                    break

            # Deserialize response
            if not data:
                logger.warning("Received empty response from H-UAV")
                return None

            response = pickle.loads(data)

            # Check success
            if not response.get('success', True):
                error_msg = response.get('error', 'Unknown error')
                logger.warning(f"H-UAV returned error: {error_msg}")
                return None

            answer = response.get('answer')
            if answer is not None:
                logger.debug(f"Received answer from H-UAV: {answer[:50]}...")
                return answer

            return None

        except socket.timeout:
            logger.error(f"Request #{self.request_counter} timed out")
            return None

        except Exception as e:
            logger.error(f"Request #{self.request_counter} failed: {e}")
            return None

        finally:
            try:
                client_socket.close()
            except:
                pass


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
    """Parse ground truth value."""
    if isinstance(ground_truth, (int, float)):
        return float(ground_truth)

    gt_str = str(ground_truth)

    if qtype == 'size':
        match = re.search(r'<size>([\d.,\s]+)</size>', gt_str)
        if not match:
            match = re.search(r'size=([\d.,\s]+)', gt_str, re.IGNORECASE)

        if match:
            numbers_str = match.group(1)
            numbers = re.findall(r'[\d.]+', numbers_str)
            if numbers:
                try:
                    return float(numbers[0])
                except ValueError:
                    pass

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


class LUAVWithHierarchicalCollaboration:
    """
    L-UAV with Hierarchical H-UAV Collaboration

    - L-UAV has basic capability (3-epoch LoRA)
    - Requests enhanced answers from H-UAV (10-epoch LoRA) when needed
    - Simple and effective collaboration
    """

    def __init__(
        self,
        model_path: str,
        lora_weights: str,
        vision_tower: Optional[str] = None,
        device: str = 'cuda:1',
        lora_rank: int = 8,
        target_layers: List[int] = None,
        huav_address: Optional[str] = None,
        confidence_threshold: float = 0.5,
        load_8bit: bool = False
    ):
        self.device = device
        self.huav_address = huav_address
        self.confidence_threshold = confidence_threshold

        # Load L-UAV model with 3-epoch LoRA
        logger.info(f"Loading L-UAV with 3-epoch LoRA from {lora_weights}...")
        self.memory_weaver = create_huav_memory_weaver(
            base_llava_path=model_path,
            device=device,
            lora_rank=lora_rank,
            target_layers=target_layers or [8, 16, 24],
            load_8bit=load_8bit,
            vision_tower=vision_tower
        )

        # Load L-UAV's 3-epoch LoRA weights
        logger.info(f"Loading L-UAV LoRA weights (3 epochs)...")
        checkpoint = torch.load(lora_weights, map_location=device)
        self.memory_weaver.load_lora_state(checkpoint)
        logger.info("✓ L-UAV LoRA loaded (3-epoch baseline)")

        self.tokenizer = self.memory_weaver.tokenizer
        self.image_processor = self.memory_weaver.image_processor

        # Create H-UAV client if address provided
        self.huav_client = None
        if self.huav_address:
            self.huav_client = LUAVSocketClient(huav_address=self.huav_address)
            logger.info(f"✓ Will request enhanced answers from H-UAV at {self.huav_address}")
            logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        else:
            logger.info("⚠️  Running in standalone mode (no H-UAV)")

        self.stats = {
            'total_queries': 0,
            'huav_requests': 0,
            'standalone_inferences': 0
        }

    def infer(
        self,
        image: Image.Image,
        question: str
    ) -> Tuple[str, float]:
        """
        Perform inference with L-UAV's 3-epoch LoRA.

        Args:
            image: Input image
            question: Question text

        Returns:
            (answer, confidence)
        """
        # Prepare image
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.to(self.device).to(torch.float16)

        # Prepare prompt
        question_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], question_with_image)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)

        # Generate with L-UAV's 3-epoch LoRA
        with torch.inference_mode():
            output_ids = self.memory_weaver.base_model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
                use_cache=True
            )

        # Decode
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if "ASSISTANT:" in outputs:
            answer = outputs.split("ASSISTANT:")[-1].strip()
        else:
            answer = outputs

        # Estimate confidence (simple heuristic)
        confidence = 0.7  # Medium confidence with 3-epoch LoRA

        return answer, confidence

    def evaluate_with_huav(
        self,
        image: Image.Image,
        question: str,
        force_huav: bool = False
    ) -> Tuple[str, float, bool]:
        """Evaluate with H-UAV collaboration."""
        self.stats['total_queries'] += 1

        # Base inference with L-UAV's 3-epoch LoRA
        answer_luav, confidence_luav = self.infer(image, question)

        # Check if H-UAV help is needed
        should_use_huav = force_huav or (confidence_luav < self.confidence_threshold)

        if self.huav_client and should_use_huav:
            if force_huav:
                logger.info(f"🔧 Forced H-UAV query (sample checkpoint)")
            else:
                logger.debug(f"Low confidence ({confidence_luav:.3f}), requesting H-UAV answer...")

            # Request answer from H-UAV (10-epoch LoRA)
            answer_huav = self.huav_client.request_answer(image, question)

            if answer_huav is not None:
                # Use H-UAV's enhanced answer
                logger.info(f"  ✅ Using H-UAV's answer (10-epoch LoRA)")
                self.stats['huav_requests'] += 1
                return answer_huav, 0.9, True  # High confidence with H-UAV

        # Use L-UAV's answer
        self.stats['standalone_inferences'] += 1
        return answer_luav, confidence_luav, False


def evaluate_luav_with_huav(
    luav: LUAVWithHierarchicalCollaboration,
    test_data: List[Dict],
    image_dir: str,
    output_path: str,
    force_huav_every_n: int = None
):
    """Evaluate L-UAV + H-UAV hierarchical collaboration."""
    print("=" * 70)
    print("L-UAV (3-epoch LoRA) + H-UAV (10-epoch LoRA) Hierarchical Collaboration V3")
    print("=" * 70)
    print(f"Total samples: {len(test_data)}")
    print(f"H-UAV address: {luav.huav_address}")
    print(f"Confidence threshold: {luav.confidence_threshold}")
    if force_huav_every_n:
        print(f"🔧 Forced H-UAV query: Every {force_huav_every_n} samples")
    print("Strategy: L-UAV requests H-UAV's answer directly (not memory tokens)")
    print("=" * 70)
    print()

    results = []
    errors = []

    for i, sample in enumerate(tqdm(test_data, desc="L-UAV + H-UAV Evaluation")):
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

            # Determine if we should force H-UAV query
            force_huav = False
            if force_huav_every_n and (i % force_huav_every_n == 0):
                force_huav = True

            # Evaluate with H-UAV collaboration
            answer, confidence, used_huav = luav.evaluate_with_huav(image, question_clean, force_huav=force_huav)

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
                'relative_error': relative_error,
                'confidence': confidence,
                'used_huav': used_huav
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

        # H-UAV usage statistics
        huav_used_count = sum(1 for r in results if r.get('used_huav', False))
        huav_usage_rate = huav_used_count / len(results) if results else 0

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
        print("L-UAV + H-UAV Hierarchical Collaboration Results")
        print("=" * 70)
        print(f"Total samples: {len(test_data)}")
        print(f"Valid predictions: {len(valid_results)}")
        print(f"Failed predictions: {len(results) - len(valid_results)}")
        print()
        print("Collaboration Statistics:")
        print(f"  L-UAV LoRA: 3 epochs (basic capability)")
        print(f"  H-UAV LoRA: 10 epochs (enhanced capability)")
        print(f"  Strategy: L-UAV requests H-UAV's answer directly")
        print(f"  H-UAV requests: {luav.stats['huav_requests']}")
        print(f"  Standalone inferences: {luav.stats['standalone_inferences']}")
        print(f"  H-UAV usage rate: {huav_usage_rate:.2%}")
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
                'collaboration_stats': {
                    'luav_lora_epochs': 3,
                    'huav_lora_epochs': 10,
                    'strategy': 'answer_delegation',  # L-UAV requests H-UAV's answer
                    'huav_requests': luav.stats['huav_requests'],
                    'standalone_inferences': luav.stats['standalone_inferences'],
                    'huav_usage_rate': huav_usage_rate
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
            'huav_usage_rate': huav_usage_rate
        }

    else:
        print("\n❌ No valid predictions!")
        return None


def main():
    parser = argparse.ArgumentParser(description='L-UAV + H-UAV Hierarchical Collaboration V3')

    # L-UAV model
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to base LLaVA model')
    parser.add_argument('--vision_tower', type=str, default=None)
    parser.add_argument('--lora_weights', type=str, required=True,
                       help='Path to L-UAV 3-epoch LoRA weights')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--load_8bit', action='store_true')

    # LoRA config
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--target_layers', type=int, nargs='+', default=[8, 16, 24])

    # H-UAV connection
    parser.add_argument('--huav_address', type=str, default='localhost:50052',
                       help='H-UAV server address (host:port)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for H-UAV requests')
    parser.add_argument('--force_huav_every_n', type=int, default=None,
                       help='Force H-UAV query every N samples')

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

    # Create L-UAV with H-UAV collaboration
    logger.info("Creating L-UAV with 3-epoch LoRA + H-UAV hierarchical collaboration...")
    luav = LUAVWithHierarchicalCollaboration(
        model_path=args.model_path,
        lora_weights=args.lora_weights,
        vision_tower=args.vision_tower,
        device=args.device,
        lora_rank=args.lora_rank,
        target_layers=args.target_layers,
        huav_address=args.huav_address,
        confidence_threshold=args.confidence_threshold,
        load_8bit=args.load_8bit
    )

    # Evaluate
    output_path = os.path.join(args.output_dir, 'results.json')
    evaluate_luav_with_huav(
        luav=luav,
        test_data=test_data,
        image_dir=args.image_dir,
        output_path=output_path,
        force_huav_every_n=args.force_huav_every_n
    )


if __name__ == '__main__':
    main()
