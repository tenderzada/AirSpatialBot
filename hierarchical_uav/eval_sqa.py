"""
SQA (Spatial Question Answering) Evaluation Script for Hierarchical UAV System

Evaluates spatial regression tasks using:
- H-UAV: Resource-rich agent with MAC memory
- L-UAV: Lightweight agent with selective querying

SQA Task Types (6 types):
1. depth - "What is the depth of the car...?"
2. distance - "How many meters is the car from the drone?"
3. length - "What is the length of the car...? (Unit: millimeter)"
4. width - "What is the width of the car...?"
5. height - "How high is the car...?"
6. size - "Output the length, width, and height..."

Evaluation Metrics: RMSE, MAE, R-squared

Usage:
    # H-UAV (on GPU 0):
    python hierarchical_uav/eval_sqa.py --uav_type h-uav --device cuda:0 --port 50051

    # L-UAV (on GPU 1):
    python hierarchical_uav/eval_sqa.py --uav_type l-uav --device cuda:1 --huav_address localhost:50051
"""

import argparse
import torch
import json
import os
import sys
import logging
import re
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_uav.models import LLaVAWithMAC, UAVConfig, UAVType
from hierarchical_uav.communication import HUAVServer, LUAVClient, SelfMatchingModule
from hierarchical_uav.dynamic_memory_weight import DynamicMemoryWeightAdjuster
from PIL import Image


def crop_image_by_bbox(image: Image.Image, bbox: List[int], padding: float = 0.1) -> Image.Image:
    """
    Crop image to the bounding box region with optional padding.

    Args:
        image: PIL Image
        bbox: [min_x, min_y, max_x, max_y]
        padding: Padding ratio to add around bbox

    Returns:
        Cropped PIL Image
    """
    img_w, img_h = image.size
    min_x, min_y, max_x, max_y = bbox

    # Add padding
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    pad_x = int(bbox_w * padding)
    pad_y = int(bbox_h * padding)

    # Clamp to image bounds
    min_x = max(0, min_x - pad_x)
    min_y = max(0, min_y - pad_y)
    max_x = min(img_w, max_x + pad_x)
    max_y = min(img_h, max_y + pad_y)

    return image.crop((min_x, min_y, max_x, max_y))


def parse_bbox_from_question(question: str) -> Optional[List[int]]:
    """
    Extract bbox coordinates from question string.

    Example: "Please tell me the depth of the car<bbox>[[2322, 870, 2428, 988]]</bbox>"

    Returns: [min_x, min_y, max_x, max_y] or None
    """
    # Try multiple bbox formats
    patterns = [
        r'<bbox>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]</bbox>',  # [[x,y,x,y]]
        r'<bbox>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</bbox>',       # [x,y,x,y]
    ]

    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return [int(match.group(i)) for i in range(1, 5)]

    return None


def parse_ground_truth(ground_truth, qtype: str) -> Optional[float]:
    """
    Parse ground truth value, handling different formats.

    Args:
        ground_truth: Ground truth value (can be float, int, or string)
        qtype: Question type (depth, distance, length, width, height, size)

    Returns:
        Numeric ground truth value or None
    """
    # If already numeric, return it
    if isinstance(ground_truth, (int, float)):
        return float(ground_truth)

    # Convert to string for parsing
    gt_str = str(ground_truth)

    # For 'size' type, handle format like '<size>4753,1832,1469</size>'
    if qtype == 'size':
        # Extract numbers from <size>...</size> format
        match = re.search(r'<size>([\d.,\s]+)</size>', gt_str)
        if match:
            numbers_str = match.group(1)
            # Split by comma and take first value (length)
            numbers = re.findall(r'[\d.]+', numbers_str)
            if numbers:
                try:
                    return float(numbers[0])
                except ValueError:
                    pass

    # For other types or fallback, try direct conversion
    try:
        return float(gt_str)
    except ValueError:
        # Try to extract first numeric value
        numbers = re.findall(r'[\d.]+', gt_str)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass

    return None


def extract_numeric_answer(text: str, qtype: str) -> Optional[float]:
    """
    Extract numeric answer from model output.

    Args:
        text: Model's text output
        qtype: Question type (depth, distance, length, width, height, size)

    Returns:
        Extracted numeric value or None
    """
    # For 'size' type, may need to extract multiple values
    if qtype == 'size':
        # Look for patterns like "length: 4500, width: 1800, height: 1600"
        # Return first numeric value for now (can be enhanced)
        numbers = re.findall(r'[\d.]+', text)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None

    # For single value types (depth, distance, length, width, height)
    # Try to find numeric values (integers or floats)
    numbers = re.findall(r'[\d.]+', text)

    if not numbers:
        return None

    # Take the first numeric value found
    try:
        return float(numbers[0])
    except ValueError:
        return None


def compute_regression_metrics(predictions: List[float], ground_truths: List[float]) -> Dict:
    """
    Compute regression metrics: RMSE, MAE, R-squared.

    Args:
        predictions: List of predicted values
        ground_truths: List of ground truth values

    Returns:
        Dictionary with metric values
    """
    pred = np.array(predictions)
    gt = np.array(ground_truths)

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred - gt))

    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))

    # R-squared
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    ss_res = np.sum((gt - pred) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'count': len(predictions)
    }


def parse_bbox_3d(sample: Dict) -> torch.Tensor:
    """
    Parse 3D bounding box from sample data.

    Args:
        sample: Data sample containing bbox info

    Returns:
        bbox_3d: [7] tensor (x, y, z, l, w, h, θ)
    """
    # Try bbox_3d field first
    if 'bbox_3d' in sample:
        bbox_data = sample['bbox_3d']
        if isinstance(bbox_data, list) and len(bbox_data) >= 7:
            return torch.tensor(bbox_data[:7], dtype=torch.float32)
        elif isinstance(bbox_data, dict):
            return torch.tensor([
                bbox_data.get('x', 0.0),
                bbox_data.get('y', 0.0),
                bbox_data.get('z', 0.0),
                bbox_data.get('l', 1.0),
                bbox_data.get('w', 1.0),
                bbox_data.get('h', 1.0),
                bbox_data.get('theta', 0.0)
            ], dtype=torch.float32)

    # Try 2D bbox, extend to 3D
    if 'bbox' in sample:
        bbox_2d = sample['bbox']
        return torch.tensor([
            bbox_2d[0], bbox_2d[1], 0.0,  # x, y, z
            bbox_2d[2] - bbox_2d[0],      # l (width)
            bbox_2d[3] - bbox_2d[1],      # w (height)
            1.0, 0.0                       # h, theta (default)
        ], dtype=torch.float32)

    # Default normalized bbox
    return torch.tensor([0.5, 0.5, 0.0, 0.1, 0.1, 1.0, 0.0], dtype=torch.float32)


def extract_image_features(
    model: LLaVAWithMAC,
    image_path: str,
    bbox_3d: torch.Tensor
) -> torch.Tensor:
    """
    Extract image features from LLaVA's vision encoder.

    Args:
        model: LLaVA model instance
        image_path: Path to the image
        bbox_3d: [7] or [batch, 7] 3D bounding box parameters

    Returns:
        features: [hidden_size] or [batch, hidden_size] image features
    """
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = model.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.to(model.config.device)

        # Extract features using vision tower
        with torch.no_grad():
            image_features = model.llava_model.get_model().get_vision_tower()(image_tensor)
            # Use mm_projector to get final feature dimension
            image_features = model.llava_model.get_model().mm_projector(image_features)

            # Average pool over spatial dimensions to get single feature vector
            if len(image_features.shape) == 3:  # [1, num_patches, hidden_size]
                image_features = image_features.mean(dim=1)  # [1, hidden_size]

        return image_features.squeeze(0)  # [hidden_size]

    except Exception as e:
        # Fallback to random features if image cannot be loaded
        logger.warning(f"Could not load image {image_path}, using random features: {e}")
        hidden_size = model.llava_model.config.hidden_size
        return torch.randn(hidden_size).to(model.config.device)


def load_sqa_data(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load SQA test data from JSONL file.

    Args:
        data_path: Path to airspatial_sqa_test.jsonl
        max_samples: Maximum number of samples to load (for testing)

    Returns:
        List of data samples
    """
    data = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line.strip()))
    return data


def run_huav_eval(
    config: UAVConfig,
    test_data: List[Dict],
    output_path: str
):
    """
    Run H-UAV evaluation with server mode.

    Args:
        config: H-UAV configuration
        test_data: List of test samples
        output_path: Path to save results
    """
    print("=" * 60)
    print("Starting H-UAV SQA Evaluation")
    print("=" * 60)

    # Load H-UAV model
    print(f"\nLoading H-UAV model on {config.device}...")
    huav_model = LLaVAWithMAC(config)
    print("✓ H-UAV model loaded")

    # Load trained memory if specified
    if hasattr(config, 'load_memory') and config.load_memory:
        print(f"\nLoading trained MAC memory from {config.load_memory}...")
        huav_model.load_mac_state(config.load_memory)
        print("✓ Trained memory loaded - H-UAV ready with learned representations")

    # Start server
    server = HUAVServer(
        huav_model=huav_model,
        host="0.0.0.0",  # Listen on all interfaces
        port=int(config.huav_address.split(':')[1]) if config.huav_address else 50051
    )
    server.start()

    print(f"\n✓ H-UAV Server started")
    print(f"  Listening on port {server.port}")
    print(f"  Waiting for L-UAV connections...")
    print(f"\nPress Ctrl+C to stop server\n")

    try:
        # Keep server running
        import time
        while True:
            time.sleep(1)

            # Print statistics periodically
            if server.total_requests > 0 and server.total_requests % 10 == 0:
                stats = server.get_statistics()
                print(f"[H-UAV Stats] Requests: {stats['total_requests']}, "
                      f"Cache hit rate: {stats['cache_hit_rate']:.2%}")

    except KeyboardInterrupt:
        print("\n\nShutting down H-UAV server...")
        server.stop()

        # Save final statistics
        final_stats = server.get_statistics()
        print("\n" + "=" * 60)
        print("H-UAV Final Statistics")
        print("=" * 60)
        print(f"Total requests served: {final_stats['total_requests']}")
        print(f"Cache hits: {final_stats['cache_hits']}")
        print(f"Cache hit rate: {final_stats['cache_hit_rate']:.2%}")

        # Save memory state
        memory_path = output_path.replace('.jsonl', '_memory.pt')
        huav_model.save_mac_state(memory_path)
        print(f"\n✓ Memory state saved to {memory_path}")


def run_luav_eval(
    config: UAVConfig,
    test_data: List[Dict],
    output_path: str,
    standalone: bool = False
):
    """
    Run L-UAV evaluation with client mode.

    Args:
        config: L-UAV configuration
        test_data: List of test samples
        output_path: Path to save results
        standalone: If True, run in standalone mode without H-UAV connection
    """
    print("=" * 60)
    if standalone:
        print("Starting L-UAV SQA Evaluation (Standalone Mode)")
        print("No H-UAV memory augmentation - baseline performance")
    else:
        print("Starting L-UAV SQA Evaluation")
    print("=" * 60)

    # Test H-UAV connection (skip in standalone mode)
    client = None
    if not standalone:
        print(f"\nConnecting to H-UAV at {config.huav_address}...")
        client = LUAVClient(huav_address=config.huav_address)

        if not client.ping_huav():
            print(f"✗ Cannot connect to H-UAV at {config.huav_address}")
            print(f"  Please ensure H-UAV server is running first!")
            return

        print(f"✓ H-UAV server is reachable")
    else:
        print(f"\n✓ Running in standalone mode (H-UAV connection disabled)")

    # Load L-UAV model
    print(f"\nLoading L-UAV model on {config.device}...")
    luav_model = LLaVAWithMAC(config)
    print("✓ L-UAV model loaded")

    # Initialize self-matching module
    sm_module = SelfMatchingModule(
        feature_dim=luav_model.llava_model.config.hidden_size,
        threshold=config.self_match_threshold
    ).to(config.device)
    print(f"✓ Self-matching module initialized (threshold={config.self_match_threshold})")

    # Initialize dynamic memory weight adjuster
    weight_adjuster = DynamicMemoryWeightAdjuster(
        base_weight=0.5,
        min_weight=0.1,
        max_weight=0.9
    )
    print(f"✓ Dynamic memory weight adjuster initialized")

    # Evaluation loop
    print(f"\nEvaluating on {len(test_data)} samples...")
    print("=" * 60)

    results = []
    local_count = 0
    remote_count = 0

    for i, sample in enumerate(tqdm(test_data, desc="L-UAV SQA Evaluation")):
        try:
            # Parse sample
            question_id = sample.get('question_id', i)
            question = sample.get('question', '')
            qtype = sample.get('qtype', 'unknown')
            image_id = sample.get('image_id', '')
            ground_truth = sample.get('ground_truth', 0.0)

            # Parse bbox from question
            bbox_2d = parse_bbox_from_question(question)
            if bbox_2d is None:
                bbox_2d = sample.get('bbox', None)

            # Parse 3D bbox for self-matching
            bbox_3d = parse_bbox_3d(sample).unsqueeze(0).to(config.device)  # [1, 7]

            # Load and process image
            image_path = os.path.join(config.image_dir, image_id) if hasattr(config, 'image_dir') else None

            if not image_path or not os.path.exists(image_path):
                if i == 0 or (i % 100 == 0):
                    logger.warning(f"Image not found: {image_path}, skipping")
                continue

            image = Image.open(image_path).convert('RGB')

            # Crop to bbox region for better focus (critical for spatial tasks)
            if bbox_2d is not None:
                image = crop_image_by_bbox(image, bbox_2d, padding=0.15)

            image_tensor = luav_model.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.to(config.device)

            # For 8-bit models, ensure dtype consistency
            vision_tower = luav_model.llava_model.get_model().get_vision_tower()
            if hasattr(vision_tower, 'dtype'):
                image_tensor = image_tensor.to(dtype=vision_tower.dtype)

            # Extract features for self-matching
            image_features = extract_image_features(luav_model, image_path, bbox_3d)
            image_features = image_features.unsqueeze(0).float()  # [1, hidden_size]

            # Generate query and compute self-matching
            with torch.no_grad():
                query, score, should_query = sm_module(image_features, bbox_3d.float())

            # Prepare question prompt for spatial regression
            from llava.conversation import conv_templates
            from llava.constants import DEFAULT_IMAGE_TOKEN

            # Clean question and create focused prompt
            question_clean = re.sub(r'<bbox>.*?</bbox>', '', question).strip()

            # Create specific prompts for each question type
            if qtype == 'depth':
                prompt_question = f"What is the depth of this vehicle in meters? Answer with just the numeric value."
            elif qtype == 'distance':
                prompt_question = f"How many meters is this vehicle from the camera/drone? Answer with just the numeric value."
            elif qtype == 'length':
                prompt_question = f"What is the length of this vehicle in millimeters? Answer with just the numeric value."
            elif qtype == 'width':
                prompt_question = f"What is the width of this vehicle in millimeters? Answer with just the numeric value."
            elif qtype == 'height':
                prompt_question = f"What is the height of this vehicle in millimeters? Answer with just the numeric value."
            elif qtype == 'size':
                prompt_question = f"What are the length, width, and height of this vehicle in millimeters? Answer with three numeric values."
            else:
                prompt_question = question_clean if question_clean else question

            # Add image token
            question_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_question}"

            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], question_with_image)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Tokenize
            input_ids = luav_model.tokenizer(prompt, return_tensors='pt')['input_ids'].to(config.device)

            if input_ids is None or input_ids.shape[0] == 0 or input_ids.shape[1] <= 1:
                logger.error(f"Sample {i}: Invalid input_ids, skipping")
                continue

            # Decision: query H-UAV or proceed locally
            memory_value = None
            cache_hit = False
            if should_query[0].item() and client is not None:
                # Query H-UAV for memory augmentation
                try:
                    value, cache_hit = client.query_huav(query[0])
                    if value is not None:
                        memory_value = value.unsqueeze(0).to(config.device).float()
                        source = "huav"
                        remote_count += 1
                    else:
                        source = "local_fallback"
                        local_count += 1
                except Exception as e:
                    logger.warning(f"H-UAV query failed: {e}, falling back to local")
                    source = "local_fallback"
                    local_count += 1
            else:
                source = "local"
                local_count += 1

            # Generate answer with optional memory augmentation
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=config.load_8bit, dtype=torch.float16):
                    if memory_value is not None:
                        # H-UAV memory available - use memory-augmented generation
                        weight_result = weight_adjuster.compute_weight(
                            self_match_score=score[0].item(),
                            huav_confidence=None,
                            luav_confidence=None
                        )
                        memory_weight = weight_result['weight']

                        if i % 10 == 0:
                            logger.info(
                                f"Sample {i}: Using H-UAV memory (weight={memory_weight:.4f}, "
                                f"score={score[0].item():.4f})"
                            )

                        output_ids = luav_model.generate_with_memory(
                            input_ids=input_ids,
                            images=image_tensor,
                            memory_features=memory_value,
                            memory_weight=memory_weight,
                            max_new_tokens=64,  # Shorter for numeric answers
                            min_new_tokens=1,
                            do_sample=False,
                            num_beams=1
                        )
                    else:
                        # No H-UAV memory - use standard generation
                        if i % 10 == 0:
                            logger.info(f"Sample {i}: Using local inference")

                        output_ids = luav_model.llava_model.generate(
                            inputs=input_ids,
                            images=image_tensor,
                            max_new_tokens=64,
                            min_new_tokens=1,
                            do_sample=False,
                            num_beams=1
                        )

            # Decode answer
            if output_ids.shape[1] <= input_ids.shape[1]:
                answer_text = luav_model.tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True
                ).strip()
            else:
                answer_text = luav_model.tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()

            # Extract numeric value from answer
            predicted_value = extract_numeric_answer(answer_text, qtype)

            # Parse ground truth value (handles various formats including <size>...</size>)
            gt_value = parse_ground_truth(ground_truth, qtype)

            # Skip if ground truth parsing failed
            if gt_value is None:
                logger.warning(f"Sample {i}: Could not parse ground_truth '{ground_truth}' for qtype '{qtype}', skipping")
                continue

            # Record result
            results.append({
                'question_id': question_id,
                'image_id': image_id,
                'question': question,
                'qtype': qtype,
                'answer_text': answer_text,
                'predicted_value': predicted_value,
                'ground_truth': gt_value,
                'self_match_score': score[0].item(),
                'source': source,
                'cache_hit': cache_hit
            })

        except Exception as e:
            import traceback
            logger.error(f"Error processing sample {i}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue

    # Save results
    print(f"\n\nSaving results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Results saved")

    # === Compute Regression Metrics ===
    print("\n" + "=" * 60)
    print("SQA Regression Evaluation")
    print("=" * 60)

    # Filter valid predictions (where numeric extraction succeeded)
    valid_results = [r for r in results if r['predicted_value'] is not None]
    invalid_count = len(results) - len(valid_results)

    print(f"\nTotal samples: {len(results)}")
    print(f"Valid predictions: {len(valid_results)} ({len(valid_results)/len(results)*100:.1f}%)")
    print(f"Invalid predictions: {invalid_count} ({invalid_count/len(results)*100:.1f}%)")

    if len(valid_results) > 0:
        # Overall metrics
        predictions = [r['predicted_value'] for r in valid_results]
        ground_truths = [r['ground_truth'] for r in valid_results]
        overall_metrics = compute_regression_metrics(predictions, ground_truths)

        print(f"\n{'Overall Metrics':^60}")
        print(f"{'-'*60}")
        print(f"  MAE (Mean Absolute Error):    {overall_metrics['mae']:>10.2f}")
        print(f"  RMSE (Root Mean Square Error): {overall_metrics['rmse']:>10.2f}")
        print(f"  R² (R-squared):                {overall_metrics['r2']:>10.4f}")
        print(f"  Sample count:                  {overall_metrics['count']:>10d}")

        # Metrics by question type
        print(f"\n{'Metrics by Question Type':^60}")
        print(f"{'-'*60}")

        qtypes = set(r['qtype'] for r in valid_results)
        for qtype in sorted(qtypes):
            qtype_results = [r for r in valid_results if r['qtype'] == qtype]
            if len(qtype_results) > 0:
                preds = [r['predicted_value'] for r in qtype_results]
                gts = [r['ground_truth'] for r in qtype_results]
                metrics = compute_regression_metrics(preds, gts)

                print(f"\n  {qtype.upper():12s} (n={metrics['count']})")
                print(f"    MAE:  {metrics['mae']:8.2f}")
                print(f"    RMSE: {metrics['rmse']:8.2f}")
                print(f"    R²:   {metrics['r2']:8.4f}")

        # Metrics by source
        print(f"\n{'Metrics by Source':^60}")
        print(f"{'-'*60}")

        sources = set(r['source'] for r in valid_results)
        for source in sorted(sources):
            source_results = [r for r in valid_results if r['source'] == source]
            if len(source_results) > 0:
                preds = [r['predicted_value'] for r in source_results]
                gts = [r['ground_truth'] for r in source_results]
                metrics = compute_regression_metrics(preds, gts)

                print(f"\n  {source.upper():15s} (n={metrics['count']})")
                print(f"    MAE:  {metrics['mae']:8.2f}")
                print(f"    RMSE: {metrics['rmse']:8.2f}")
                print(f"    R²:   {metrics['r2']:8.4f}")

    # Print statistics
    print("\n" + "=" * 60)
    print("L-UAV Decision Statistics")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"Local decisions: {local_count} ({local_count/len(results)*100:.1f}%)")
    print(f"Remote queries: {remote_count} ({remote_count/len(results)*100:.1f}%)")

    # Cache statistics
    if remote_count > 0:
        cache_hits = sum(1 for r in results if r.get('cache_hit', False))
        cache_hit_rate = cache_hits / remote_count
        print(f"\nCache statistics:")
        print(f"  Total cache hits: {cache_hits}")
        print(f"  Cache hit rate: {cache_hit_rate:.2%}")

    # Self-matching statistics
    sm_stats = sm_module.get_statistics()
    print(f"\nSelf-matching statistics:")
    print(f"  Query rate: {sm_stats['query_rate']:.2%}")
    print(f"  Score mean: {sm_stats['score_mean']:.4f}")
    print(f"  Score std:  {sm_stats['score_std']:.4f}")

    # Show sample results
    print(f"\n" + "=" * 60)
    print("Sample Results (first 5)")
    print("=" * 60)
    for i, result in enumerate(results[:5]):
        print(f"\n[Sample {i+1}] {result['qtype']}")
        print(f"  Question: {result['question'][:70]}...")
        print(f"  Predicted: {result['predicted_value']}")
        print(f"  Ground Truth: {result['ground_truth']}")
        print(f"  Error: {abs(result['predicted_value'] - result['ground_truth']) if result['predicted_value'] else 'N/A'}")
        print(f"  Source: {result['source']}")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical UAV SQA Evaluation")

    # UAV type
    parser.add_argument(
        '--uav_type',
        type=str,
        choices=['h-uav', 'l-uav'],
        required=True,
        help='UAV type: h-uav (server) or l-uav (client)'
    )

    # Model configuration
    parser.add_argument('--model_path', type=str, default='./models/AirSpatialBot')
    parser.add_argument('--vision_tower', type=str, default='/mnt/data/clip-vit-large-patch14-336')
    parser.add_argument('--device', type=str, default='cuda:0')

    # Communication configuration
    parser.add_argument('--port', type=int, default=50051, help='Server port (H-UAV)')
    parser.add_argument('--huav_address', type=str, default='localhost:50051', help='H-UAV address (L-UAV)')
    parser.add_argument('--threshold', type=float, default=0.7, help='Self-matching threshold (L-UAV)')
    parser.add_argument(
        '--standalone',
        action='store_true',
        help='Run L-UAV in standalone mode (no H-UAV connection)'
    )

    # Data configuration
    parser.add_argument(
        '--test_data',
        type=str,
        default='./data/metadata/airspatial_sqa_test.jsonl',
        help='Path to SQA test data'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='./data/images',
        help='Directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for results'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (for testing)'
    )

    # MAC configuration
    parser.add_argument('--memory_dim', type=int, default=4096)
    parser.add_argument('--num_persistent_tokens', type=int, default=64)
    parser.add_argument('--num_memory_tokens', type=int, default=128)
    parser.add_argument(
        '--load-memory',
        type=str,
        default=None,
        help='Load trained MAC memory from checkpoint (H-UAV only)'
    )

    # Quantization options
    parser.add_argument('--load_8bit', action='store_true', help='Use 8-bit quantization')
    parser.add_argument('--load_4bit', action='store_true', help='Use 4-bit quantization')

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = f'./outputs/hierarchical_uav/sqa_{args.uav_type}_results.jsonl'

    # Load test data
    print(f"Loading SQA test data from {args.test_data}...")
    test_data = load_sqa_data(args.test_data, max_samples=args.max_samples)
    print(f"✓ Loaded {len(test_data)} test samples")

    # Create configuration based on UAV type
    if args.uav_type == 'h-uav':
        config = UAVConfig.create_huav_config(
            model_path=args.model_path,
            vision_tower=args.vision_tower,
            device=args.device,
            memory_dim=args.memory_dim,
            num_persistent_tokens=args.num_persistent_tokens,
            num_memory_tokens=args.num_memory_tokens,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit
        )
        config.huav_address = f"0.0.0.0:{args.port}"
        config.image_dir = args.image_dir
        config.load_memory = args.load_memory

        run_huav_eval(config, test_data, args.output)

    else:  # l-uav
        config = UAVConfig.create_luav_config(
            model_path=args.model_path,
            vision_tower=args.vision_tower,
            huav_address=args.huav_address,
            device=args.device,
            memory_dim=args.memory_dim,
            num_persistent_tokens=args.num_persistent_tokens,
            num_memory_tokens=args.num_memory_tokens,
            self_match_threshold=args.threshold,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit
        )
        config.image_dir = args.image_dir

        if args.standalone:
            print(f"\n⚠️  STANDALONE MODE: No H-UAV connection")
            print(f"   → All samples use local inference only\n")

        run_luav_eval(config, test_data, args.output, standalone=args.standalone)


if __name__ == "__main__":
    main()
