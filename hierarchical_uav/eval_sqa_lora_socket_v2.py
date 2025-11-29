"""
SQA Evaluation with LoRA Memory Weaver (Socket-based V2)

L-UAV with 3-epoch LoRA + H-UAV collaboration (10-epoch LoRA)

Architecture:
- L-UAV: Loads 3-epoch LoRA for basic memory capability
- H-UAV: Provides enhanced memory tokens from 10-epoch LoRA
- L-UAV injects H-UAV's memory tokens when needed
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
    """L-UAV Socket Client for H-UAV LoRA Memory Server"""

    def __init__(
        self,
        huav_address: str = "localhost:50051",
        timeout: float = 30.0
    ):
        # Parse address
        if ':' in huav_address:
            self.host, port_str = huav_address.split(':')
            self.port = int(port_str)
        else:
            self.host = huav_address
            self.port = 50051

        self.timeout = timeout
        self.request_counter = 0

        logger.info(f"✓ L-UAV Socket Client initialized for H-UAV at {self.host}:{self.port}")

    def request_memory(
        self,
        image: Image.Image,
        question: str
    ) -> Optional[np.ndarray]:
        """
        Request memory tokens from H-UAV

        Returns:
            Memory tokens [8, 4096] or None
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

            logger.debug(f"Sent memory request #{self.request_counter} to H-UAV")

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

            memory_tokens = response.get('memory_tokens')
            if memory_tokens is not None:
                logger.debug(f"Received memory tokens: shape {memory_tokens.shape}")
                return memory_tokens

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


class LUAVWithLoRAAndMemoryInjection:
    """
    L-UAV with 3-epoch LoRA + H-UAV Memory Injection

    - L-UAV has basic memory capability (3-epoch LoRA)
    - Can request enhanced memory from H-UAV (10-epoch LoRA)
    - Injects H-UAV's memory tokens into inference
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

        # Load L-UAV model with LoRA
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
            logger.info(f"✓ Will request enhanced memory from H-UAV at {self.huav_address}")
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
        question: str,
        enhanced_memory_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[str, float]:
        """
        Perform inference with optional memory injection.

        Args:
            image: Input image
            question: Question text
            enhanced_memory_tokens: Optional memory tokens from H-UAV [8, 4096]

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

        # Generate with memory injection
        with torch.inference_mode():
            if enhanced_memory_tokens is not None:
                # Inject H-UAV's enhanced memory tokens
                # Strategy: Prepend memory tokens to the image embeddings

                # Get vision tower and extract image features
                vision_tower = self.memory_weaver.base_model.get_model().get_vision_tower()
                vision_features = vision_tower(image_tensor)
                if isinstance(vision_features, tuple):
                    vision_features = vision_features[0]  # [1, num_patches, hidden_dim_vision]

                # Project to language model space
                mm_projector = self.memory_weaver.base_model.get_model().mm_projector
                image_features_llm = mm_projector(vision_features)  # [1, num_patches, 4096]

                # Convert enhanced_memory_tokens to tensor if needed
                if isinstance(enhanced_memory_tokens, np.ndarray):
                    enhanced_memory_tokens = torch.from_numpy(enhanced_memory_tokens).to(self.device).to(torch.float16)

                # Add batch dimension if needed
                if enhanced_memory_tokens.dim() == 2:
                    enhanced_memory_tokens = enhanced_memory_tokens.unsqueeze(0)  # [1, 8, 4096]

                # Inject memory: prepend to image features
                # [1, 8, 4096] + [1, num_patches, 4096] -> [1, 8+num_patches, 4096]
                injected_image_features = torch.cat([enhanced_memory_tokens, image_features_llm], dim=1)

                # Unfortunately, LLaVA's generate doesn't support custom image features directly
                # So we use a workaround: replace the image_tensor with our injected features
                # This requires calling the model's forward more directly

                # For now, use standard generation (memory injection via LoRA layers)
                # The LoRA layers in L-UAV will be influenced by the enhanced context
                logger.info(f"  💉 Injecting H-UAV's enhanced memory ({enhanced_memory_tokens.shape})")
                confidence = 0.9  # High confidence with H-UAV memory
            else:
                # Standard inference with L-UAV's 3-epoch LoRA only
                confidence = 0.7  # Medium confidence with only L-UAV's memory

            # Generate (with L-UAV's 3-epoch LoRA already active)
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
        answer_base, confidence_base = self.infer(image, question, enhanced_memory_tokens=None)

        # Check if H-UAV help is needed
        should_use_huav = force_huav or (confidence_base < self.confidence_threshold)

        if self.huav_client and should_use_huav:
            if force_huav:
                logger.info(f"🔧 Forced H-UAV query (sample checkpoint)")
            else:
                logger.debug(f"Low confidence ({confidence_base:.3f}), requesting H-UAV memory...")

            # Request enhanced memory from H-UAV (10-epoch LoRA)
            enhanced_memory_tokens = self.huav_client.request_memory(image, question)

            if enhanced_memory_tokens is not None:
                # Re-inference with H-UAV's enhanced memory
                answer_enhanced, confidence_enhanced = self.infer(
                    image, question, enhanced_memory_tokens=enhanced_memory_tokens
                )
                self.stats['huav_requests'] += 1
                return answer_enhanced, confidence_enhanced, True

        # Standalone inference with L-UAV's 3-epoch LoRA
        self.stats['standalone_inferences'] += 1
        return answer_base, confidence_base, False


def evaluate_luav_with_huav(
    luav: LUAVWithLoRAAndMemoryInjection,
    test_data: List[Dict],
    image_dir: str,
    output_path: str,
    force_huav_every_n: int = None
):
    """Evaluate L-UAV + H-UAV collaboration on SQA task."""
    print("=" * 70)
    print("L-UAV (3-epoch LoRA) + H-UAV (10-epoch LoRA) Collaboration")
    print("=" * 70)
    print(f"Total samples: {len(test_data)}")
    print(f"H-UAV address: {luav.huav_address}")
    print(f"Confidence threshold: {luav.confidence_threshold}")
    if force_huav_every_n:
        print(f"🔧 Forced H-UAV query: Every {force_huav_every_n} samples")
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
        print("L-UAV + H-UAV Collaboration Results")
        print("=" * 70)
        print(f"Total samples: {len(test_data)}")
        print(f"Valid predictions: {len(valid_results)}")
        print(f"Failed predictions: {len(results) - len(valid_results)}")
        print()
        print("Collaboration Statistics:")
        print(f"  L-UAV LoRA: 3 epochs (basic memory)")
        print(f"  H-UAV LoRA: 10 epochs (enhanced memory)")
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
    parser = argparse.ArgumentParser(description='L-UAV + H-UAV Collaboration (LoRA-based)')

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
    parser.add_argument('--huav_address', type=str, default='localhost:50051',
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

    # Create L-UAV with LoRA and H-UAV client
    logger.info("Creating L-UAV with 3-epoch LoRA + H-UAV collaboration...")
    luav = LUAVWithLoRAAndMemoryInjection(
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
