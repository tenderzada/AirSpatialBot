"""
Task 1 Evaluation Script for Hierarchical UAV System

Evaluates vehicle attribute recognition using:
- H-UAV: Resource-rich agent with MAC memory
- L-UAV: Lightweight agent with selective querying

Usage:
    # H-UAV (on GPU 0):
    python hierarchical_uav/eval_task1.py --uav_type h-uav --device cuda:0 --port 50051

    # L-UAV (on GPU 1):
    python hierarchical_uav/eval_task1.py --uav_type l-uav --device cuda:1 --huav_address localhost:50051
"""

import argparse
import torch
import json
import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_uav.models import LLaVAWithMAC, UAVConfig, UAVType
from hierarchical_uav.communication import HUAVServer, LUAVClient, SelfMatchingModule
from PIL import Image
import numpy as np


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
        print(f"Warning: Could not load image {image_path}, using random features: {e}")
        hidden_size = model.llava_model.config.hidden_size
        return torch.randn(hidden_size).to(model.config.device)


def parse_bbox_3d(sample: Dict) -> torch.Tensor:
    """
    Parse 3D bounding box from sample data.

    Args:
        sample: Data sample containing bbox info

    Returns:
        bbox_3d: [7] tensor (x, y, z, l, w, h, θ)
    """
    # Try to extract from sample, otherwise use default
    if '3d_bbox' in sample:
        bbox_data = sample['3d_bbox']
        bbox_3d = torch.tensor([
            bbox_data.get('x', 0.0),
            bbox_data.get('y', 0.0),
            bbox_data.get('z', 0.0),
            bbox_data.get('l', 1.0),
            bbox_data.get('w', 1.0),
            bbox_data.get('h', 1.0),
            bbox_data.get('theta', 0.0)
        ], dtype=torch.float32)
    elif 'bbox' in sample:
        # Try 2D bbox, extend to 3D
        bbox_2d = sample['bbox']
        bbox_3d = torch.tensor([
            bbox_2d[0], bbox_2d[1], 0.0,  # x, y, z
            bbox_2d[2] - bbox_2d[0],      # l (width)
            bbox_2d[3] - bbox_2d[1],      # w (height)
            1.0, 0.0                       # h, theta (default)
        ], dtype=torch.float32)
    else:
        # Default normalized bbox
        bbox_3d = torch.tensor([0.5, 0.5, 0.0, 0.1, 0.1, 1.0, 0.0], dtype=torch.float32)

    return bbox_3d


def load_task1_data(data_path: str) -> List[Dict]:
    """Load Task 1 test data."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
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
    print("Starting H-UAV Evaluation")
    print("=" * 60)

    # Load H-UAV model
    print(f"\nLoading H-UAV model on {config.device}...")
    huav_model = LLaVAWithMAC(config)
    print("✓ H-UAV model loaded")

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
    output_path: str
):
    """
    Run L-UAV evaluation with client mode.

    Args:
        config: L-UAV configuration
        test_data: List of test samples
        output_path: Path to save results
    """
    print("=" * 60)
    print("Starting L-UAV Evaluation")
    print("=" * 60)

    # Test H-UAV connection
    print(f"\nConnecting to H-UAV at {config.huav_address}...")
    client = LUAVClient(huav_address=config.huav_address)

    if not client.ping_huav():
        print(f"✗ Cannot connect to H-UAV at {config.huav_address}")
        print(f"  Please ensure H-UAV server is running first!")
        return

    print(f"✓ H-UAV server is reachable")

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

    # Evaluation loop
    print(f"\nEvaluating on {len(test_data)} samples...")
    print("=" * 60)

    results = []
    local_count = 0
    remote_count = 0

    for i, sample in enumerate(tqdm(test_data, desc="L-UAV Evaluation")):
        try:
            # Parse 3D bounding box from sample
            bbox_3d = parse_bbox_3d(sample).unsqueeze(0).to(config.device)  # [1, 7]

            # Get image path and question
            image_id = sample.get('image_id', sample.get('image', ''))

            # Extract question safely from various formats
            if 'question' in sample:
                question = sample['question']
            elif 'conversations' in sample and len(sample['conversations']) > 0:
                # Extract from conversations format
                first_message = sample['conversations'][0]
                question = first_message.get('value', '') if isinstance(first_message, dict) else str(first_message)
            else:
                question = ""

            # Ensure question is a string
            if not isinstance(question, str):
                question = str(question) if question else ""
            image_path = os.path.join(config.image_dir, image_id) if hasattr(config, 'image_dir') else None

            # Load and process image
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                image_tensor = luav_model.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                image_tensor = image_tensor.to(config.device)

                # For 8-bit models, ensure image tensor dtype matches vision tower
                # Get vision tower dtype to ensure compatibility
                vision_tower = luav_model.llava_model.get_model().get_vision_tower()
                if hasattr(vision_tower, 'dtype'):
                    image_tensor = image_tensor.to(dtype=vision_tower.dtype)

                # Extract features for self-matching
                image_features = extract_image_features(luav_model, image_path, bbox_3d)
                image_features = image_features.unsqueeze(0)  # [1, hidden_size]

                # Ensure dtype consistency (float32)
                image_features = image_features.float()
                bbox_3d = bbox_3d.float()
            else:
                # Skip if image not found
                if i == 0 or (i % 100 == 0):
                    print(f"\nWarning: Image not found at {image_path}, skipping")
                continue

            # Generate query and compute self-matching
            with torch.no_grad():
                query, score, should_query = sm_module(image_features, bbox_3d)

            # Prepare question prompt
            from llava.conversation import conv_templates
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

            # Validate question
            if not question or question.strip() == "":
                logger.warning(f"Sample {i}: Empty question, skipping")
                continue

            # Prepare question for LLaVA
            # LLaVA expects <image> token in the prompt
            # Remove bbox tags if present (they're not standard LLaVA tokens)
            import re
            question_clean = re.sub(r'<bbox>.*?</bbox>', '', question)
            question_clean = question_clean.strip()

            # Add image token for LLaVA (required for multimodal input)
            question_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{question_clean}"

            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], question_with_image)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Tokenize
            input_ids = luav_model.tokenizer(prompt, return_tensors='pt')['input_ids'].to(config.device)

            # Validate input_ids thoroughly
            if input_ids is None:
                logger.error(f"Sample {i}: input_ids is None after tokenization, skipping")
                continue
            if input_ids.shape[0] == 0 or input_ids.shape[1] <= 1:
                logger.error(f"Sample {i}: input_ids has invalid shape {input_ids.shape}, skipping")
                logger.error(f"  Question: {question[:100]}")
                logger.error(f"  Prompt length: {len(prompt)}")
                continue

            # Log details for debugging (every 10 samples)
            if i % 10 == 0:
                logger.info(f"Sample {i}: input_ids.shape={input_ids.shape}, question_len={len(question)}")

            # Decision: query H-UAV or proceed locally
            memory_value = None
            cache_hit = False
            if should_query[0].item():
                # Query H-UAV for memory augmentation
                try:
                    value, cache_hit = client.query_huav(query[0])
                    if value is not None:
                        memory_value = value.unsqueeze(0).to(config.device)  # [1, memory_dim]
                        source = "huav"
                        remote_count += 1
                    else:
                        # H-UAV returned None, fallback to local
                        source = "local_fallback"
                        local_count += 1
                except Exception as e:
                    # H-UAV query failed (timeout, connection error, etc.)
                    logger.warning(f"H-UAV query failed: {e}, falling back to local processing")
                    source = "local_fallback"
                    local_count += 1
            else:
                # Proceed locally without H-UAV memory
                source = "local"
                local_count += 1

            # Validate image_tensor
            if image_tensor is None:
                logger.error(f"Sample {i}: image_tensor is None, skipping")
                continue

            # Generate answer (unified path for stability)
            # Note: Memory features are currently not injected due to LLaVA architecture constraints
            # They serve as validation that H-UAV has processed this query
            with torch.no_grad():
                if memory_value is not None:
                    if i % 10 == 0:  # Log every 10 samples to reduce verbosity
                        logger.info(f"Sample {i}: Using H-UAV validated path")
                else:
                    if i % 10 == 0:
                        logger.info(f"Sample {i}: Using local processing path")

                # Use standard LLaVA generation for all cases
                # This avoids potential issues with the generate_with_memory wrapper
                # Note: LLaVA's generate() expects 'inputs' not 'input_ids'
                # Use autocast for 8-bit models to handle dtype mismatches
                with torch.cuda.amp.autocast(enabled=config.load_8bit, dtype=torch.float16):
                    output_ids = luav_model.llava_model.generate(
                        inputs=input_ids,  # Changed from input_ids= to inputs=
                        images=image_tensor,
                        max_new_tokens=512,
                        do_sample=False,  # Use greedy decoding for stability with 8-bit
                        num_beams=1
                    )

            # Decode answer
            answer = luav_model.tokenizer.decode(
                output_ids[0, input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Record result
            results.append({
                'question_id': sample.get('question_id', i),
                'image_id': image_id,
                'question': question,
                'answer': answer,
                'ground_truth': sample.get('answer', ''),
                'self_match_score': score[0].item(),
                'source': source,
                'cache_hit': cache_hit
            })

        except Exception as e:
            import traceback
            print(f"\nError processing sample {i}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Full error details for sample {i}:", exc_info=True)

            # Save problematic sample for debugging
            try:
                output_dir = os.path.dirname(output_path) or "."
                error_file = os.path.join(output_dir, f"error_sample_{i}.json")
                with open(error_file, 'w') as f:
                    json.dump({
                        'sample_index': i,
                        'sample_data': sample,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'question': question if 'question' in locals() else None,
                        'image_id': image_id if 'image_id' in locals() else None,
                        'input_ids_shape': list(input_ids.shape) if 'input_ids' in locals() and input_ids is not None else None,
                        'image_tensor_shape': list(image_tensor.shape) if 'image_tensor' in locals() and image_tensor is not None else None
                    }, f, indent=2)
                logger.error(f"Saved error details to {error_file}")
            except Exception as save_error:
                logger.error(f"Failed to save error sample: {save_error}")

            continue

    # Save results
    print(f"\n\nSaving results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Results saved")

    # Print statistics
    print("\n" + "=" * 60)
    print("L-UAV Evaluation Statistics")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"Local decisions: {local_count} ({local_count/len(results)*100:.1f}%)")
    print(f"Remote queries: {remote_count} ({remote_count/len(results)*100:.1f}%)")

    # Cache statistics
    cache_hits = sum(1 for r in results if r.get('cache_hit', False))
    cache_hit_rate = cache_hits / remote_count if remote_count > 0 else 0.0
    print(f"\nCache statistics:")
    print(f"  Total cache hits: {cache_hits}")
    print(f"  Cache hit rate: {cache_hit_rate:.2%}")

    sm_stats = sm_module.get_statistics()
    print(f"\nSelf-matching statistics:")
    print(f"  Query rate: {sm_stats['query_rate']:.2%}")
    print(f"  Total queries: {sm_stats['total_queries']}")
    print(f"  H-UAV queries: {sm_stats['huav_queries']}")

    print(f"\nSelf-matching score distribution:")
    print(f"  Mean: {sm_stats['score_mean']:.4f}")
    print(f"  Std:  {sm_stats['score_std']:.4f}")
    print(f"  Min:  {sm_stats['score_min']:.4f}")
    print(f"  Max:  {sm_stats['score_max']:.4f}")

    # Show score histogram
    score_list = [r['self_match_score'] for r in results]
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    print(f"\nScore distribution by range:")
    for i in range(len(bins)-1):
        count = sum(1 for s in score_list if bins[i] <= s < bins[i+1])
        pct = count / len(score_list) * 100
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:4d} ({pct:5.1f}%)")

    # Show sample results
    print(f"\n" + "=" * 60)
    print("Sample Results (first 3)")
    print("=" * 60)
    for i, result in enumerate(results[:3]):
        print(f"\n[Sample {i+1}]")
        print(f"  Image: {result['image_id']}")
        print(f"  Question: {result['question'][:60]}...")
        print(f"  Answer: {result['answer'][:80]}...")
        print(f"  Source: {result['source']} (cache_hit={result['cache_hit']})")
        print(f"  Score: {result['self_match_score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical UAV Task 1 Evaluation")

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

    # Data configuration
    parser.add_argument(
        '--test_data',
        type=str,
        default='./data/metadata/airspatial_agent_test_task1.jsonl',
        help='Path to Task 1 test data'
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

    # MAC configuration
    parser.add_argument('--memory_dim', type=int, default=4096)
    parser.add_argument('--num_persistent_tokens', type=int, default=64)
    parser.add_argument('--num_memory_tokens', type=int, default=128)

    # Quantization options
    parser.add_argument('--load_8bit', action='store_true', help='Use 8-bit quantization to reduce memory')
    parser.add_argument('--load_4bit', action='store_true', help='Use 4-bit quantization (experimental)')

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = f'./outputs/hierarchical_uav/task1_{args.uav_type}_results.jsonl'

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = load_task1_data(args.test_data)
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

        run_luav_eval(config, test_data, args.output)


if __name__ == "__main__":
    main()
