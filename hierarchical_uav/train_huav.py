"""
H-UAV Test-Time Learning Script

Performs continual learning on H-UAV to build meaningful memory representations.
This is Phase 3 of the hierarchical UAV system.

Usage:
    python hierarchical_uav/train_huav.py \
        --model_path /mnt/data/AirSpatialBot \
        --train_data ./data/metadata/airspatial_agent_test_task1.jsonl \
        --output_dir ./outputs/huav_training \
        --device cuda:0 \
        --num_epochs 3

The script will:
1. Load H-UAV model with MAC layers
2. Process training samples sequentially (test-time learning)
3. Update MAC memory based on surprise-driven learning
4. Save memory checkpoints periodically
5. Generate training report
"""

import argparse
import torch
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_uav.models import LLaVAWithMAC, UAVConfig, UAVType
from PIL import Image
import numpy as np
import re


def parse_bbox_3d(sample: Dict) -> torch.Tensor:
    """Parse 3D bounding box from sample data."""
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
        bbox_2d = sample['bbox']
        bbox_3d = torch.tensor([
            bbox_2d[0], bbox_2d[1], 0.0,
            bbox_2d[2] - bbox_2d[0],
            bbox_2d[3] - bbox_2d[1],
            1.0, 0.0
        ], dtype=torch.float32)
    else:
        bbox_3d = torch.tensor([0.5, 0.5, 0.0, 0.1, 0.1, 1.0, 0.0], dtype=torch.float32)

    return bbox_3d


def load_training_data(data_path: str) -> List[Dict]:
    """Load training data from JSONL file."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def validate_data(train_data: List[Dict], image_dir: str) -> Dict:
    """
    Validate training data and return statistics.

    Returns dict with:
        - total: Total samples
        - has_question: Samples with questions
        - has_image: Samples with existing images
        - valid: Samples that have both
    """
    stats = {
        'total': len(train_data),
        'has_question': 0,
        'has_image': 0,
        'valid': 0,
        'sample_keys': set()
    }

    for i, sample in enumerate(train_data[:100]):  # Check first 100
        # Collect all keys to understand data structure
        stats['sample_keys'].update(sample.keys())

        # Check question
        has_q = False
        if 'question' in sample and sample['question']:
            has_q = True
        elif 'conversations' in sample and len(sample['conversations']) > 0:
            has_q = True
        if has_q:
            stats['has_question'] += 1

        # Check image
        image_id = sample.get('image_id', sample.get('image', ''))
        if image_id:
            image_path = os.path.join(image_dir, image_id)
            if os.path.exists(image_path):
                stats['has_image'] += 1
                if has_q:
                    stats['valid'] += 1

    stats['sample_keys'] = list(stats['sample_keys'])
    return stats


def train_huav(
    model: LLaVAWithMAC,
    train_data: List[Dict],
    config: UAVConfig,
    args
):
    """
    Perform test-time learning on H-UAV.

    Args:
        model: H-UAV model with MAC layers
        train_data: List of training samples
        config: UAV configuration
        args: Training arguments
    """

    logger.info("=" * 70)
    logger.info("H-UAV Test-Time Learning (Phase 3)")
    logger.info("=" * 70)
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Num epochs: {args.num_epochs}")
    logger.info(f"Device: {config.device}")
    logger.info(f"MAC config:")
    logger.info(f"  - Memory dim: {config.memory_dim}")
    logger.info(f"  - Persistent tokens: {config.num_persistent_tokens}")
    logger.info(f"  - Working memory tokens: {config.num_memory_tokens}")
    logger.info(f"  - Learning rate (theta): {config.learning_theta}")
    logger.info(f"  - Surprise decay (eta): {config.surprise_eta}")
    logger.info(f"  - Forgetting rate (alpha): {config.forgetting_alpha}")
    logger.info("=" * 70)

    # Training statistics
    total_samples = 0
    total_loss = 0.0
    total_surprise = 0.0
    epoch_stats = []

    # Ensure model is in eval mode (test-time learning, not training)
    model.eval()

    # Process each epoch
    for epoch in range(args.num_epochs):
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"{'='*70}")

        epoch_loss = 0.0
        epoch_surprise = 0.0
        epoch_samples = 0
        high_surprise_count = 0

        # Progress bar
        pbar = tqdm(train_data, desc=f"Epoch {epoch+1}")

        # Track skip reasons for debugging
        skip_reasons = {
            'no_question': 0,
            'no_image': 0,
            'errors': 0
        }

        for i, sample in enumerate(pbar):
            try:
                # Get image path and question
                image_id = sample.get('image_id', sample.get('image', ''))

                # Extract question
                if 'question' in sample:
                    question = sample['question']
                elif 'conversations' in sample and len(sample['conversations']) > 0:
                    question = sample['conversations'][0].get('value', '')
                else:
                    skip_reasons['no_question'] += 1
                    if i < 5:  # Log first few skips
                        logger.warning(f"Sample {i}: No question found. Keys: {list(sample.keys())}")
                    continue

                if not question or question.strip() == "":
                    skip_reasons['no_question'] += 1
                    continue

                # Load image
                image_path = os.path.join(args.image_dir, image_id)
                if not os.path.exists(image_path):
                    skip_reasons['no_image'] += 1
                    if i < 5:  # Log first few skips
                        logger.warning(f"Sample {i}: Image not found at {image_path}")
                    continue

                image = Image.open(image_path).convert('RGB')
                image_tensor = model.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                image_tensor = image_tensor.to(config.device)

                # Prepare prompt
                from llava.conversation import conv_templates
                from llava.constants import DEFAULT_IMAGE_TOKEN

                question_clean = re.sub(r'<bbox>.*?</bbox>', '', question).strip()
                question_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{question_clean}"

                conv = conv_templates["vicuna_v1"].copy()
                conv.append_message(conv.roles[0], question_with_image)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = model.tokenizer(prompt, return_tensors='pt')['input_ids'].to(config.device)

                # Forward pass with memory update enabled
                # IMPORTANT: Do NOT use torch.no_grad() for test-time learning!
                # Memory update requires gradients for surprise-driven learning
                outputs = model(
                    input_ids=input_ids,
                    images=image_tensor,
                    update_memory=True  # Enable test-time learning!
                )

                # Extract metrics
                memory_metrics = outputs.get('memory_metrics', {})
                loss = memory_metrics.get('memory_loss', 0.0)
                surprise = memory_metrics.get('surprise', 0.0)

                # Update statistics
                epoch_loss += loss
                epoch_surprise += surprise
                epoch_samples += 1
                total_samples += 1

                # Track high surprise samples (novelty)
                if abs(surprise) > 0.5:
                    high_surprise_count += 1

                # Update progress bar
                if epoch_samples > 0:
                    avg_loss = epoch_loss / epoch_samples
                    avg_surprise = epoch_surprise / epoch_samples
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'surprise': f'{avg_surprise:.4f}',
                        'novel': f'{high_surprise_count}'
                    })

                # Periodic logging
                if (i + 1) % args.log_interval == 0:
                    logger.info(
                        f"  Sample {i+1}/{len(train_data)}: "
                        f"loss={loss:.4f}, surprise={surprise:.4f}"
                    )

                # Clear CUDA cache periodically to prevent memory fragmentation
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                skip_reasons['errors'] += 1
                if i < 5:  # Log first few errors
                    logger.warning(f"Error processing sample {i}: {e}")
                continue

        # Epoch summary
        avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        avg_epoch_surprise = epoch_surprise / epoch_samples if epoch_samples > 0 else 0.0

        epoch_stats.append({
            'epoch': epoch + 1,
            'samples': epoch_samples,
            'avg_loss': avg_epoch_loss,
            'avg_surprise': avg_epoch_surprise,
            'high_surprise_count': high_surprise_count
        })

        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"  Samples processed: {epoch_samples}")

        # Report skip reasons if samples were skipped
        total_skipped = skip_reasons['no_question'] + skip_reasons['no_image'] + skip_reasons['errors']
        if total_skipped > 0:
            logger.warning(f"  Samples skipped: {total_skipped}")
            logger.warning(f"    - No question: {skip_reasons['no_question']}")
            logger.warning(f"    - Image not found: {skip_reasons['no_image']}")
            logger.warning(f"    - Errors: {skip_reasons['errors']}")

        if epoch_samples > 0:
            logger.info(f"  Average loss: {avg_epoch_loss:.4f}")
            logger.info(f"  Average surprise: {avg_epoch_surprise:.4f}")
            logger.info(f"  High surprise samples: {high_surprise_count} ({high_surprise_count/epoch_samples*100:.1f}%)")
        else:
            logger.error(f"  ⚠️  NO SAMPLES PROCESSED!")
            logger.error(f"  Please check:")
            logger.error(f"    1. Training data path: {args.train_data}")
            logger.error(f"    2. Image directory: {args.image_dir}")
            logger.error(f"    3. Data format (see logs above)")
            break  # Stop training if no samples processed

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.output_dir,
                f'huav_memory_epoch_{epoch+1}.pt'
            )
            save_checkpoint(model, checkpoint_path, epoch + 1, epoch_stats)
            logger.info(f"  ✓ Checkpoint saved: {checkpoint_path}")

    # Final statistics
    logger.info(f"\n{'='*70}")
    if total_samples > 0:
        logger.info("Training Complete!")
    else:
        logger.error("Training Failed - No Samples Processed!")
    logger.info(f"{'='*70}")
    logger.info(f"Total samples processed: {total_samples}")
    logger.info(f"Total epochs completed: {len(epoch_stats)}/{args.num_epochs}")

    # Only save if we processed samples
    if total_samples == 0:
        logger.error("\n❌ Training failed - no samples were processed!")
        logger.error("\nCommon issues:")
        logger.error("  1. Image directory path is incorrect")
        logger.error("  2. Data file format doesn't match expected structure")
        logger.error("  3. All images are missing from the directory")
        logger.error("\nPlease check the warning messages above for details.")
        return epoch_stats

    # Save final model
    final_path = os.path.join(args.output_dir, 'huav_memory_final.pt')
    save_checkpoint(model, final_path, args.num_epochs, epoch_stats)
    logger.info(f"\n✓ Final memory saved: {final_path}")

    # Save training report
    report_path = os.path.join(args.output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump({
            'config': {
                'model_path': args.model_path,
                'train_data': args.train_data,
                'num_epochs': args.num_epochs,
                'memory_dim': config.memory_dim,
                'learning_theta': config.learning_theta,
                'surprise_eta': config.surprise_eta,
                'forgetting_alpha': config.forgetting_alpha
            },
            'results': {
                'total_samples': total_samples,
                'epoch_stats': epoch_stats
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"✓ Training report saved: {report_path}")

    return epoch_stats


def save_checkpoint(model, path, epoch, stats):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save MAC state
    if hasattr(model, 'mac_layer'):
        checkpoint = {
            'epoch': epoch,
            'mac_state': model.mac_layer.state_dict(),
            'neural_memory_state': model.mac_layer.neural_memory.state_dict(),
            'persistent_memory_state': model.mac_layer.persistent_memory.state_dict(),
            'stats': stats
        }
        torch.save(checkpoint, path)


def load_checkpoint(model, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=model.config.device)

    if hasattr(model, 'mac_layer'):
        model.mac_layer.load_state_dict(checkpoint['mac_state'])
        logger.info(f"✓ Loaded MAC checkpoint from epoch {checkpoint['epoch']}")

    return checkpoint.get('stats', [])


def main():
    parser = argparse.ArgumentParser(description="H-UAV Test-Time Learning")

    # Model configuration
    parser.add_argument('--model_path', type=str, default='./models/AirSpatialBot')
    parser.add_argument('--vision_tower', type=str, default='/mnt/data/clip-vit-large-patch14-336')
    parser.add_argument('--device', type=str, default='cuda:0')

    # Data configuration
    parser.add_argument('--train_data', type=str,
                       default='./data/metadata/airspatial_agent_test_task1.jsonl')
    parser.add_argument('--image_dir', type=str, default='./data/images')

    # Training configuration
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Log every N samples')
    parser.add_argument('--save_interval', type=int, default=1,
                       help='Save checkpoint every N epochs')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./outputs/huav_training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    # MAC configuration
    parser.add_argument('--memory_dim', type=int, default=4096)
    parser.add_argument('--num_persistent_tokens', type=int, default=64)
    parser.add_argument('--num_memory_tokens', type=int, default=128)
    parser.add_argument('--learning_theta', type=float, default=0.1,
                       help='Learning rate for memory update')
    parser.add_argument('--surprise_eta', type=float, default=0.9,
                       help='Surprise decay (momentum)')
    parser.add_argument('--forgetting_alpha', type=float, default=0.01,
                       help='Forgetting rate (weight decay)')

    # Quantization
    parser.add_argument('--load_8bit', action='store_true')
    parser.add_argument('--load_4bit', action='store_true')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load training data
    logger.info(f"Loading training data from {args.train_data}...")
    train_data = load_training_data(args.train_data)
    logger.info(f"✓ Loaded {len(train_data)} training samples")

    # Validate data before training
    logger.info(f"\nValidating data (checking first 100 samples)...")
    validation_stats = validate_data(train_data, args.image_dir)
    logger.info(f"  Sample keys found: {validation_stats['sample_keys']}")
    logger.info(f"  Samples with questions: {validation_stats['has_question']}/100")
    logger.info(f"  Samples with images: {validation_stats['has_image']}/100")
    logger.info(f"  Valid samples (both): {validation_stats['valid']}/100")

    if validation_stats['valid'] == 0:
        logger.error("\n❌ No valid samples found!")
        logger.error("Please check:")
        logger.error(f"  1. Image directory exists: {args.image_dir}")
        logger.error(f"  2. Data file format matches expected structure")
        logger.error(f"  3. Sample has required keys: {validation_stats['sample_keys']}")
        return

    logger.info(f"✓ Data validation passed")

    # Create H-UAV configuration
    config = UAVConfig.create_huav_config(
        model_path=args.model_path,
        vision_tower=args.vision_tower,
        device=args.device,
        memory_dim=args.memory_dim,
        num_persistent_tokens=args.num_persistent_tokens,
        num_memory_tokens=args.num_memory_tokens,
        learning_theta=args.learning_theta,
        surprise_eta=args.surprise_eta,
        forgetting_alpha=args.forgetting_alpha,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit
    )

    # Set image directory
    args.image_dir = args.image_dir if hasattr(args, 'image_dir') else './data/images'

    # Load H-UAV model
    logger.info(f"\nLoading H-UAV model on {config.device}...")
    huav_model = LLaVAWithMAC(config)

    # Enable gradient checkpointing to reduce memory usage during training
    # This is critical for training on 24GB GPUs with 7B models
    if hasattr(huav_model.llava_model, 'enable_input_require_grads'):
        huav_model.llava_model.enable_input_require_grads()
    if hasattr(huav_model.llava_model.model, 'gradient_checkpointing_enable'):
        huav_model.llava_model.model.gradient_checkpointing_enable()
        logger.info("✓ Enabled gradient checkpointing (memory-efficient training)")

    logger.info("✓ H-UAV model loaded")

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        load_checkpoint(huav_model, args.resume)

    # Run training
    train_huav(huav_model, train_data, config, args)

    logger.info("\n" + "=" * 70)
    logger.info("H-UAV Training Complete!")
    logger.info("=" * 70)
    logger.info(f"\nNext steps:")
    logger.info(f"1. Review training report: {args.output_dir}/training_report.json")
    logger.info(f"2. Start H-UAV server with trained memory:")
    logger.info(f"   ./run_huav.sh --load-memory {args.output_dir}/huav_memory_final.pt")
    logger.info(f"3. Evaluate L-UAV with trained H-UAV memory")


if __name__ == "__main__":
    main()
