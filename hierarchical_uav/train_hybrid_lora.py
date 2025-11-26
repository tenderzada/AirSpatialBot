"""
Hybrid MAC-LoRA Training Script for SQA Task

Trains H-UAV with hybrid memory architecture combining:
- MAC: Surprise-driven test-time learning
- LoRA: Parameter-efficient memory generation
- Adaptive Trigger: Learns when to invoke memory

Usage:
    python hierarchical_uav/train_hybrid_lora.py \
        --model_path /mnt/data/AirSpatialBot \
        --train_data /mnt/data/AirSpatial/airspatial_sqa_test.jsonl \
        --output_dir ./outputs/hybrid_lora_training_sqa \
        --device cuda:0 \
        --training_mode hybrid
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import logging
from datetime import datetime
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_uav.models import LLaVAWithMAC, UAVConfig, UAVType
from hierarchical_uav.mac_memory import HybridMACLoRALayer, HybridMACLoRAConfig
from PIL import Image
import numpy as np


def parse_ground_truth(ground_truth, qtype: str) -> Optional[float]:
    """
    Parse ground truth value from various formats.

    Handles:
    - Direct numeric values
    - XML-like tags: <size>x,y,z</size>
    - Strings with embedded numbers
    """
    if isinstance(ground_truth, (int, float)):
        return float(ground_truth)

    gt_str = str(ground_truth)

    # Handle size type with XML format
    if qtype == 'size':
        match = re.search(r'<size>([\d.,\s]+)</size>', gt_str)
        if match:
            numbers = re.findall(r'[\d.]+', match.group(1))
            if numbers:
                return float(numbers[0])  # Return length (first dimension)

    # Try direct conversion
    try:
        return float(gt_str)
    except ValueError:
        # Extract first number from string
        numbers = re.findall(r'[\d.]+', gt_str)
        if numbers:
            return float(numbers[0])

    return None


def load_training_data(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load training data from JSONL file."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            data.append(sample)

            if max_samples and len(data) >= max_samples:
                break

    return data


def validate_sqa_data(train_data: List[Dict], image_dir: str) -> Dict:
    """Validate SQA training data and return statistics."""
    stats = {
        'total': len(train_data),
        'valid_question': 0,
        'valid_image': 0,
        'valid_gt': 0,
        'valid_complete': 0,
        'qtype_counts': {}
    }

    for sample in train_data[:100]:  # Check first 100
        # Check question
        if 'question' in sample and sample['question']:
            stats['valid_question'] += 1

        # Check image
        image_id = sample.get('image_id', sample.get('image', ''))
        if image_id:
            image_path = os.path.join(image_dir, image_id)
            if os.path.exists(image_path):
                stats['valid_image'] += 1

        # Check ground truth
        qtype = sample.get('question_type', 'unknown')
        gt = sample.get('ground_truth')
        if gt is not None:
            parsed_gt = parse_ground_truth(gt, qtype)
            if parsed_gt is not None:
                stats['valid_gt'] += 1

        # Count question types
        stats['qtype_counts'][qtype] = stats['qtype_counts'].get(qtype, 0) + 1

        # Count complete samples
        if stats['valid_question'] > 0 and stats['valid_image'] > 0 and stats['valid_gt'] > 0:
            stats['valid_complete'] += 1

    return stats


def create_hybrid_model(args, base_model_path: str, device: str) -> LLaVAWithMAC:
    """
    Create H-UAV model with hybrid MAC-LoRA architecture.

    Returns:
        Model with hybrid memory layer
    """
    # Create base config
    config = UAVConfig(
        uav_type=UAVType.H_UAV,
        device=device,
        # MAC parameters
        memory_dim=4096,
        memory_depth=2,
        num_persistent_tokens=64,
        num_memory_tokens=32,
        # Learning parameters
        learning_theta=args.learning_theta,
        surprise_eta=args.surprise_eta,
        forgetting_alpha=args.forgetting_alpha,
        # Model paths
        model_path=base_model_path,
        vision_tower=args.vision_tower,
        load_8bit=args.load_8bit
    )

    # Create model with standard MAC
    model = LLaVAWithMAC(config)

    # Replace MAC layer with Hybrid MAC-LoRA layer
    hybrid_config = HybridMACLoRAConfig(
        hidden_size=4096,
        num_attention_heads=32,
        memory_dim=4096,
        memory_depth=2,
        num_persistent_tokens=64,
        # LoRA memory
        num_lora_memory_tokens=10,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        # MAC memory
        num_mac_memory_tokens=32,
        enable_mac_memory=True,
        # Trigger
        enable_trigger=args.enable_trigger,
        trigger_threshold=args.trigger_threshold,
        # Learning
        enable_memory_update=True,
        surprise_eta=args.surprise_eta,
        learning_theta=args.learning_theta,
        forgetting_alpha=args.forgetting_alpha,
        # Fusion
        fusion_mode=args.fusion_mode
    )

    # Replace the MAC layer
    model.mac_layer = HybridMACLoRALayer(hybrid_config).to(device)

    logger.info("Created Hybrid MAC-LoRA model")
    logger.info(f"  Fusion mode: {args.fusion_mode}")
    logger.info(f"  LoRA rank: {args.lora_rank}")
    logger.info(f"  Adaptive trigger: {args.enable_trigger}")

    return model


def train_hybrid_lora(
    model: LLaVAWithMAC,
    train_data: List[Dict],
    args
):
    """
    Train hybrid MAC-LoRA model on SQA task.

    Training modes:
        - hybrid: Train both MAC and LoRA
        - lora_only: Train only LoRA parameters
        - mac_only: Train only MAC memory (original behavior)
    """

    logger.info("=" * 70)
    logger.info("Hybrid MAC-LoRA Training")
    logger.info("=" * 70)
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Num epochs: {args.num_epochs}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Training mode: {args.training_mode}")
    logger.info(f"Fusion mode: {args.fusion_mode}")
    logger.info("=" * 70)

    # Set up optimizer for LoRA parameters
    optimizer = None
    if args.training_mode in ['hybrid', 'lora_only']:
        lora_params = list(model.mac_layer.get_lora_parameters())
        optimizer = optim.AdamW(lora_params, lr=args.lora_lr, weight_decay=0.01)
        logger.info(f"LoRA optimizer: AdamW with lr={args.lora_lr}")
        logger.info(f"LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    # Training statistics
    total_samples = 0
    total_loss = 0.0
    total_surprise = 0.0
    total_trigger_invocations = 0
    epoch_stats = []

    # Set model to eval mode (for test-time learning)
    model.eval()

    # Process each epoch
    for epoch in range(args.num_epochs):
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"{'='*70}")

        epoch_loss = 0.0
        epoch_surprise = 0.0
        epoch_samples = 0
        epoch_trigger_invokes = 0
        high_surprise_count = 0

        # Progress bar
        pbar = tqdm(train_data, desc=f"Epoch {epoch+1}")

        skip_reasons = {
            'no_question': 0,
            'no_image': 0,
            'no_gt': 0,
            'errors': 0
        }

        for i, sample in enumerate(pbar):
            try:
                # Get image and question
                image_id = sample.get('image_id', sample.get('image', ''))
                question = sample.get('question', '')
                qtype = sample.get('question_type', 'unknown')
                ground_truth = sample.get('ground_truth')

                # Validate sample
                if not question:
                    skip_reasons['no_question'] += 1
                    continue

                if not image_id:
                    skip_reasons['no_image'] += 1
                    continue

                # Parse ground truth
                gt_value = parse_ground_truth(ground_truth, qtype)
                if gt_value is None:
                    skip_reasons['no_gt'] += 1
                    continue

                # Load image
                image_path = os.path.join(args.image_dir, image_id)
                if not os.path.exists(image_path):
                    skip_reasons['no_image'] += 1
                    continue

                image = Image.open(image_path).convert('RGB')

                # Process through model
                # Note: For SQA, we don't use 3D bbox
                with torch.no_grad() if args.training_mode == 'lora_only' else torch.enable_grad():
                    # Forward pass through hybrid model
                    output_ids = model.llava_model.generate(
                        image,
                        question,
                        max_new_tokens=512,
                        temperature=0.2
                    )

                    # Get MAC layer metrics
                    if hasattr(model, 'last_mac_metrics'):
                        metrics = model.last_mac_metrics

                        # Update statistics
                        if 'memory_loss' in metrics:
                            epoch_loss += metrics['memory_loss']
                        if 'surprise' in metrics:
                            epoch_surprise += metrics['surprise']
                            if metrics['surprise'] > 0.5:
                                high_surprise_count += 1
                        if 'trigger_invoke_rate' in metrics:
                            epoch_trigger_invokes += metrics['trigger_invoke_rate']

                epoch_samples += 1
                total_samples += 1

                # Update LoRA parameters
                if optimizer is not None and epoch_samples % args.log_interval == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # Update progress bar
                if epoch_samples > 0:
                    avg_loss = epoch_loss / epoch_samples
                    avg_surprise = epoch_surprise / epoch_samples
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'surprise': f'{avg_surprise:.4f}',
                        'high_surprise': high_surprise_count
                    })

                # Log detailed statistics
                if (i + 1) % args.log_interval == 0:
                    logger.info(f"\n[Epoch {epoch+1}] Sample {i+1}/{len(train_data)}")
                    logger.info(f"  Avg loss: {avg_loss:.4f}")
                    logger.info(f"  Avg surprise: {avg_surprise:.4f}")
                    logger.info(f"  High surprise samples: {high_surprise_count}")
                    if args.enable_trigger:
                        avg_trigger = epoch_trigger_invokes / max(epoch_samples, 1)
                        logger.info(f"  Trigger invoke rate: {avg_trigger:.2%}")

            except Exception as e:
                skip_reasons['errors'] += 1
                if skip_reasons['errors'] <= 5:
                    logger.warning(f"Error processing sample {i}: {str(e)}")
                continue

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(epoch_samples, 1)
        avg_epoch_surprise = epoch_surprise / max(epoch_samples, 1)

        epoch_stat = {
            'epoch': epoch + 1,
            'samples_processed': epoch_samples,
            'avg_loss': avg_epoch_loss,
            'avg_surprise': avg_epoch_surprise,
            'high_surprise_samples': high_surprise_count,
            'skip_reasons': skip_reasons
        }

        if args.enable_trigger:
            epoch_stat['trigger_invoke_rate'] = epoch_trigger_invokes / max(epoch_samples, 1)

        epoch_stats.append(epoch_stat)

        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch+1} Summary:")
        logger.info(f"  Processed: {epoch_samples} / {len(train_data)}")
        logger.info(f"  Avg loss: {avg_epoch_loss:.4f}")
        logger.info(f"  Avg surprise: {avg_epoch_surprise:.4f}")
        logger.info(f"  High surprise: {high_surprise_count}")
        logger.info(f"  Skipped - no_question: {skip_reasons['no_question']}")
        logger.info(f"  Skipped - no_image: {skip_reasons['no_image']}")
        logger.info(f"  Skipped - no_gt: {skip_reasons['no_gt']}")
        logger.info(f"  Skipped - errors: {skip_reasons['errors']}")
        logger.info("=" * 70)

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, args.output_dir, epoch + 1, args.training_mode)

    # Save final model
    save_checkpoint(model, args.output_dir, 'final', args.training_mode)

    # Generate training report
    report = {
        'training_mode': args.training_mode,
        'fusion_mode': args.fusion_mode,
        'total_samples': total_samples,
        'num_epochs': args.num_epochs,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'enable_trigger': args.enable_trigger,
        'epoch_stats': epoch_stats,
        'config': {
            'learning_theta': args.learning_theta,
            'surprise_eta': args.surprise_eta,
            'forgetting_alpha': args.forgetting_alpha,
            'lora_lr': args.lora_lr if optimizer else None
        },
        'timestamp': datetime.now().isoformat()
    }

    report_path = os.path.join(args.output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nTraining report saved to: {report_path}")

    return report


def save_checkpoint(model: LLaVAWithMAC, output_dir: str, epoch, training_mode: str):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)

    # Save hybrid memory state
    memory_path = os.path.join(output_dir, f'hybrid_memory_epoch{epoch}.pt')
    model.save_mac_state(memory_path)
    logger.info(f"Saved hybrid memory to: {memory_path}")

    # Save LoRA weights separately
    if training_mode in ['hybrid', 'lora_only']:
        lora_state = {
            'memory_weaver': model.mac_layer.memory_weaver.state_dict(),
        }
        if model.mac_layer.memory_trigger is not None:
            lora_state['memory_trigger'] = model.mac_layer.memory_trigger.state_dict()

        lora_path = os.path.join(output_dir, f'lora_weights_epoch{epoch}.pt')
        torch.save(lora_state, lora_path)
        logger.info(f"Saved LoRA weights to: {lora_path}")


def main():
    parser = argparse.ArgumentParser(description='Hybrid MAC-LoRA Training for SQA')

    # Model paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to base LLaVA model')
    parser.add_argument('--vision_tower', type=str, required=True,
                       help='Path to vision tower')

    # Data paths
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data (JSONL)')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to use (for testing)')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints')

    # Device
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--load_8bit', action='store_true',
                       help='Use 8-bit quantization')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=1,
                       help='Checkpoint save interval (epochs)')

    # MAC parameters
    parser.add_argument('--learning_theta', type=float, default=0.1,
                       help='MAC learning rate')
    parser.add_argument('--surprise_eta', type=float, default=0.9,
                       help='Surprise decay')
    parser.add_argument('--forgetting_alpha', type=float, default=0.01,
                       help='Forgetting rate')

    # LoRA parameters
    parser.add_argument('--lora_rank', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=16.0,
                       help='LoRA alpha (scaling)')
    parser.add_argument('--lora_lr', type=float, default=1e-4,
                       help='LoRA learning rate')

    # Hybrid parameters
    parser.add_argument('--fusion_mode', type=str, default='concat',
                       choices=['concat', 'add', 'learned', 'adaptive'],
                       help='Memory fusion mode')
    parser.add_argument('--enable_trigger', action='store_true',
                       help='Enable adaptive triggering')
    parser.add_argument('--trigger_threshold', type=float, default=0.5,
                       help='Trigger threshold')

    # Training mode
    parser.add_argument('--training_mode', type=str, default='hybrid',
                       choices=['hybrid', 'lora_only', 'mac_only'],
                       help='Training mode')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and validate data
    logger.info("Loading training data...")
    train_data = load_training_data(args.train_data, args.max_samples)
    logger.info(f"Loaded {len(train_data)} samples")

    logger.info("Validating data...")
    stats = validate_sqa_data(train_data, args.image_dir)
    logger.info("Data validation stats:")
    for key, value in stats.items():
        if key != 'qtype_counts':
            logger.info(f"  {key}: {value}")
    logger.info(f"  Question types: {stats['qtype_counts']}")

    # Create hybrid model
    logger.info("Creating hybrid MAC-LoRA model...")
    model = create_hybrid_model(args, args.model_path, args.device)

    # Train
    logger.info("Starting training...")
    report = train_hybrid_lora(model, train_data, args)

    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Total samples processed: {report['total_samples']}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
