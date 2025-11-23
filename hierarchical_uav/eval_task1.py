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
from hierarchical_uav.dynamic_memory_weight import DynamicMemoryWeightAdjuster
from PIL import Image
import numpy as np
import pandas as pd
import re


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


def parse_bbox_from_question(question: str) -> List[int]:
    """Extract bbox coordinates from question string."""
    match = re.search(r'<bbox>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</bbox>', question)
    if match:
        return [int(match.group(i)) for i in range(1, 5)]
    return None


class VehicleKnowledgeBase:
    """Knowledge base for vehicle attributes lookup based on 3D size."""

    def __init__(self, csv_files: List[str] = None):
        self.df = None
        if csv_files:
            self.load_csv(csv_files)

    def load_csv(self, csv_files: List[str]):
        """Load vehicle data from CSV files."""
        dfs = []
        for f in csv_files:
            if os.path.exists(f):
                dfs.append(pd.read_csv(f))
        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(self.df)} vehicle records from knowledge base")

    def query_by_size(self, length: float, width: float, height: float, tolerance: float = 50.0) -> Dict:
        """Query vehicle attributes by 3D dimensions with tolerance."""
        if self.df is None:
            return None

        # Find matching rows within tolerance
        mask = (
            (abs(self.df['length'] - length) <= tolerance) &
            (abs(self.df['width'] - width) <= tolerance) &
            (abs(self.df['height'] - height) <= tolerance)
        )
        matches = self.df[mask]

        if len(matches) == 0:
            return None

        # Return first match
        row = matches.iloc[0]
        return {
            'brand': str(row.get('brand', '')),
            'model': str(row.get('model', '')),
            'type': str(row.get('type', '')),
            'color': str(row.get('color', '')),
            'min_price': row.get('min_price', row.get('price', None)),
            'powertrain': str(row.get('powertrain', ''))
        }

    def query_by_image_bbox(self, image_id: str, bbox: List[int]) -> Dict:
        """Query vehicle attributes by image ID and bbox."""
        if self.df is None or 'Image URL' not in self.df.columns:
            return None

        # Match by image and bbox
        mask = (
            (self.df['Image URL'] == image_id) &
            (self.df['x_min'] == bbox[0]) &
            (self.df['y_min'] == bbox[1]) &
            (self.df['x_max'] == bbox[2]) &
            (self.df['y_max'] == bbox[3])
        )
        matches = self.df[mask]

        if len(matches) == 0:
            return None

        row = matches.iloc[0]
        return {
            'brand': str(row.get('brand', '')),
            'model': str(row.get('model', '')),
            'type': str(row.get('type', '')),
            'color': str(row.get('color', '')),
            'min_price': row.get('min_price', row.get('price', None)),
            'powertrain': str(row.get('powertrain', '')),
            'length': row.get('length', 0),
            'width': row.get('width', 0),
            'height': row.get('height', 0)
        }


def get_question_type(question: str) -> str:
    """Determine the type of question being asked."""
    question_lower = question.lower()
    if 'color' in question_lower or 'colour' in question_lower:
        return 'color'
    elif 'type' in question_lower:
        return 'type'
    elif 'brand' in question_lower:
        return 'brand'
    elif 'model' in question_lower:
        return 'model'
    elif 'price' in question_lower or 'cost' in question_lower:
        return 'min_price'
    elif 'powertrain' in question_lower or 'fuel' in question_lower or 'electric' in question_lower:
        return 'powertrain'
    return 'unknown'


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
        print("Starting L-UAV Evaluation (Standalone Mode)")
        print("No H-UAV memory augmentation - baseline performance")
    else:
        print("Starting L-UAV Evaluation")
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

    # Initialize dynamic memory weight adjuster (Phase 1 optimization)
    weight_adjuster = DynamicMemoryWeightAdjuster(
        base_weight=0.5,
        min_weight=0.1,
        max_weight=0.9
    )
    print(f"✓ Dynamic memory weight adjuster initialized")
    print(f"  Strategy: Adaptive weighting based on self-match scores")
    print(f"  Range: [{weight_adjuster.min_weight}, {weight_adjuster.max_weight}]")

    # Initialize vehicle knowledge base (like main_task1.py)
    knowledge_base = VehicleKnowledgeBase()
    csv_files = [
        "csv_file/output-19-add.csv",
        "csv_file/output-20-add.csv",
        "csv_file/output-40-add.csv",
        "csv_file/output-42-add.csv",
        "csv_file/output-43-add.csv",
        "csv_file/output-44-add.csv",
        "csv_file/output-45-add.csv",
        "csv_file/output-46-add.csv",
        "csv_file/output-47-add.csv",
        "csv_file/output-48-add.csv",
        "csv_file/output-49-add.csv",
    ]
    knowledge_base.load_csv(csv_files)
    if knowledge_base.df is not None:
        print(f"✓ Vehicle knowledge base loaded ({len(knowledge_base.df)} records)")
    else:
        print("⚠ Vehicle knowledge base not found, using VLM-only mode")

    # Evaluation loop
    print(f"\nEvaluating on {len(test_data)} samples...")
    print("=" * 60)

    results = []
    local_count = 0
    remote_count = 0
    forced_query_count = 0  # Track samples forced to query H-UAV

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

            # Validate question
            if not question or question.strip() == "":
                logger.warning(f"Sample {i}: Empty question, skipping")
                continue

            # Parse bbox from question (like main_task1.py)
            bbox_2d = parse_bbox_from_question(question)
            if bbox_2d is None:
                # Try from sample data
                bbox_2d = sample.get('bbox', None)

            # Determine question type
            qtype = get_question_type(question)

            # === Research Mode: Force Query Rate ===
            # Randomly force some samples to query H-UAV for testing memory mechanism
            import random
            force_query_huav = random.random() < config.force_query_rate
            if force_query_huav:
                forced_query_count += 1

            # === Strategy: Use knowledge base for brand/model/price/type/powertrain ===
            # Use VLM for color (requires visual understanding)
            kb_answer = None
            if knowledge_base.df is not None and bbox_2d is not None and not force_query_huav:
                # Try to get answer from knowledge base (like main_task1.py)
                # Skip KB lookup if forced to query H-UAV (for research)
                kb_result = knowledge_base.query_by_image_bbox(image_id, bbox_2d)
                if kb_result and qtype in kb_result and kb_result[qtype]:
                    kb_answer = str(kb_result[qtype])
                    if kb_answer and kb_answer != 'nan' and kb_answer != '':
                        # Found answer in knowledge base
                        results.append({
                            'question_id': sample.get('question_id', i),
                            'image_id': image_id,
                            'question': question,
                            'answer': kb_answer,
                            'ground_truth': sample.get('gt', sample.get('answer', '')),
                            'qtype': qtype,
                            'self_match_score': 0.0,
                            'source': 'knowledge_base',
                            'cache_hit': False
                        })
                        local_count += 1
                        continue

            # === Fallback to VLM inference ===
            image_path = os.path.join(config.image_dir, image_id) if hasattr(config, 'image_dir') else None

            # Load and process image
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')

                # Crop to bbox region for better focus (like img_slice in main_task1.py)
                if bbox_2d is not None:
                    image = crop_image_by_bbox(image, bbox_2d, padding=0.15)

                image_tensor = luav_model.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                image_tensor = image_tensor.to(config.device)

                # For 8-bit models, ensure image tensor dtype matches vision tower
                vision_tower = luav_model.llava_model.get_model().get_vision_tower()
                if hasattr(vision_tower, 'dtype'):
                    image_tensor = image_tensor.to(dtype=vision_tower.dtype)

                # Extract features for self-matching (use original image for consistency)
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

            # Clean question and create better prompt for vehicle attribute recognition
            question_clean = re.sub(r'<bbox>.*?</bbox>', '', question).strip()

            # Create focused prompt based on question type
            if qtype == 'color':
                prompt_question = "What is the color of this car in the image? Answer with just the color name."
            elif qtype == 'type':
                prompt_question = "What type/class is this car (e.g., sedan, SUV, hatchback, mid-size, compact)? Answer briefly."
            elif qtype == 'brand':
                prompt_question = "What brand/make is this car? Answer with just the brand name."
            elif qtype == 'model':
                prompt_question = "What model is this car? Answer with just the model name."
            else:
                prompt_question = question_clean if question_clean else question

            # Add image token for LLaVA (required for multimodal input)
            question_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_question}"

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
            if should_query[0].item() and client is not None:
                # Query H-UAV for memory augmentation (only if not in standalone mode)
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
                # Proceed locally without H-UAV memory (or in standalone mode)
                source = "local"
                local_count += 1

            # Validate image_tensor
            if image_tensor is None:
                logger.error(f"Sample {i}: image_tensor is None, skipping")
                continue

            # Generate answer with optional memory augmentation
            with torch.no_grad():
                # Use autocast for 8-bit models to handle dtype mismatches
                with torch.cuda.amp.autocast(enabled=config.load_8bit, dtype=torch.float16):
                    if memory_value is not None:
                        # H-UAV memory available - use memory-augmented generation

                        # Convert memory_value to the correct device and dtype
                        memory_value = memory_value.to(config.device).float()

                        # PHASE 1 OPTIMIZATION: Dynamic memory weight based on self-match score
                        # Compute adaptive weight instead of using fixed 0.5
                        weight_result = weight_adjuster.compute_weight(
                            self_match_score=score[0].item(),
                            huav_confidence=None,  # TODO: Can be added from H-UAV response
                            luav_confidence=None   # TODO: Can be extracted from L-UAV logits
                        )
                        memory_weight = weight_result['weight']

                        # Log decision for analysis
                        if i % 10 == 0:  # Log every 10 samples to reduce verbosity
                            decision_summary = weight_adjuster.get_decision_summary(memory_weight)
                            logger.info(
                                f"Sample {i}: Using H-UAV memory-augmented inference\n"
                                f"  Self-match score: {score[0].item():.4f}\n"
                                f"  Memory weight: {memory_weight:.4f}\n"
                                f"  Strategy: {weight_result['strategy']}\n"
                                f"  Decision: {decision_summary}"
                            )

                        # Call generate_with_memory to inject H-UAV memory features
                        output_ids = luav_model.generate_with_memory(
                            input_ids=input_ids,
                            images=image_tensor,
                            memory_features=memory_value,
                            memory_weight=memory_weight,  # ADAPTIVE WEIGHT (was fixed 0.5)
                            max_new_tokens=512,
                            min_new_tokens=1,
                            do_sample=False,
                            num_beams=1
                        )
                    else:
                        # No H-UAV memory - use standard generation
                        if i % 10 == 0:
                            logger.info(f"Sample {i}: Using local inference without H-UAV memory")

                        output_ids = luav_model.llava_model.generate(
                            inputs=input_ids,
                            images=image_tensor,
                            max_new_tokens=512,
                            min_new_tokens=1,
                            do_sample=False,
                            num_beams=1
                        )

            # Decode answer
            # Note: LLaVA generate() may return only generated tokens (not input+output)
            # Check if output is shorter than input - if so, decode entire output
            if output_ids.shape[1] <= input_ids.shape[1]:
                # Model returned only generated tokens
                answer = luav_model.tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True
                ).strip()
            else:
                # Model returned input + generated tokens
                answer = luav_model.tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()

            # Record result
            gt = sample.get('gt', sample.get('answer', ''))
            results.append({
                'question_id': sample.get('question_id', i),
                'image_id': image_id,
                'question': question,
                'answer': answer,
                'ground_truth': str(gt),
                'qtype': qtype,
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
    if forced_query_count > 0:
        print(f"  ↳ Forced queries (research mode): {forced_query_count} ({forced_query_count/len(results)*100:.1f}%)")
        natural_query_count = remote_count - forced_query_count
        if natural_query_count < 0:
            natural_query_count = 0
        print(f"  ↳ Natural queries (self-matching): {natural_query_count} ({natural_query_count/len(results)*100:.1f}%)")

    # Cache statistics
    cache_hits = sum(1 for r in results if r.get('cache_hit', False))
    cache_hit_rate = cache_hits / remote_count if remote_count > 0 else 0.0
    print(f"\nCache statistics:")
    print(f"  Total cache hits: {cache_hits}")
    print(f"  Cache hit rate: {cache_hit_rate:.2%}")

    # Dynamic memory weight statistics (Phase 1 optimization)
    print(f"\n" + "=" * 60)
    print("Dynamic Memory Weight Statistics (Phase 1)")
    print("=" * 60)
    print(f"Strategy: Adaptive weighting based on self-match scores")
    print(f"Weight range: [{weight_adjuster.min_weight}, {weight_adjuster.max_weight}]")
    print(f"\nExpected behavior:")
    print(f"  • High self-match (>0.7) → Higher weight (0.6-0.8)")
    print(f"  • Medium self-match (0.5-0.7) → Moderate weight (0.3-0.6)")
    print(f"  • Low self-match (<0.5) → Low weight (0.1-0.3)")
    print(f"\nNote: With current untrained self-matching, most scores will be ~0.5")
    print(f"      After Phase 3 (training), scores will become more discriminative")

    # === Accuracy Evaluation (like main_task1.py) ===
    print(f"\n" + "=" * 60)
    print("Accuracy Evaluation")
    print("=" * 60)

    correct = 0
    total = len(results)
    correct_by_type = {}
    total_by_type = {}
    correct_by_source = {'knowledge_base': 0, 'huav': 0, 'local': 0, 'local_fallback': 0}
    total_by_source = {'knowledge_base': 0, 'huav': 0, 'local': 0, 'local_fallback': 0}

    for r in results:
        gt = str(r.get('ground_truth', '')).lower().strip()
        answer = str(r.get('answer', '')).lower().strip()
        qtype = r.get('qtype', 'unknown')
        source = r.get('source', 'unknown')

        # Count by type
        total_by_type[qtype] = total_by_type.get(qtype, 0) + 1
        if source in total_by_source:
            total_by_source[source] += 1

        # Check if correct (gt in answer, like main_task1.py)
        is_correct = gt in answer if gt else False
        if is_correct:
            correct += 1
            correct_by_type[qtype] = correct_by_type.get(qtype, 0) + 1
            if source in correct_by_source:
                correct_by_source[source] += 1

    print(f"\nOverall Accuracy: {correct}/{total} = {correct/total*100:.2f}%")

    print(f"\nAccuracy by Question Type:")
    for qtype in sorted(total_by_type.keys()):
        type_correct = correct_by_type.get(qtype, 0)
        type_total = total_by_type[qtype]
        print(f"  {qtype:12s}: {type_correct:3d}/{type_total:3d} = {type_correct/type_total*100:5.1f}%")

    print(f"\nAccuracy by Source:")
    for source in ['knowledge_base', 'huav', 'local', 'local_fallback']:
        if total_by_source[source] > 0:
            src_correct = correct_by_source[source]
            src_total = total_by_source[source]
            print(f"  {source:15s}: {src_correct:3d}/{src_total:3d} = {src_correct/src_total*100:5.1f}%")

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
    parser.add_argument(
        '--force-query-rate',
        type=float,
        default=0.0,
        help='Force this percentage of samples to query H-UAV (0.0-1.0). '
             'For research: bypass KB lookup for X%% of samples to test memory mechanism. '
             'Example: 0.3 = force 30%% of samples to query H-UAV'
    )
    parser.add_argument(
        '--standalone',
        action='store_true',
        help='Run L-UAV in standalone mode (no H-UAV connection). '
             'Uses only local inference and knowledge base. '
             'Useful for baseline performance evaluation.'
    )

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
    parser.add_argument(
        '--load-memory',
        type=str,
        default=None,
        help='Load trained MAC memory from checkpoint (H-UAV only). '
             'Example: ./outputs/huav_training/huav_memory_final.pt'
    )

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
        config.load_memory = args.load_memory  # Path to trained memory checkpoint

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
        config.force_query_rate = args.force_query_rate

        # Log research mode settings
        if args.standalone:
            print(f"\n⚠️  STANDALONE MODE: No H-UAV connection")
            print(f"   → All samples use local inference and knowledge base only\n")
        elif args.force_query_rate > 0:
            print(f"\n⚠️  RESEARCH MODE: Force query rate = {args.force_query_rate*100:.1f}%")
            print(f"   → {args.force_query_rate*100:.1f}% of samples will bypass KB and test H-UAV memory\n")

        run_luav_eval(config, test_data, args.output, standalone=args.standalone)


if __name__ == "__main__":
    main()
