"""
Training script for MemGen-style Memory Weaver

This trains the learnable query latents and LoRA adapters to generate
context-aware memory tokens for H-UAV.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hierarchical_uav.mac_memory.memgen_weaver import MemGenWeaver, MemGenWeaverConfig


class MemoryWeaverDataset(Dataset):
    """
    Dataset for training MemGen Weaver.

    Each sample contains:
    - image + question (encoded as hidden states)
    - ground truth answer
    """

    def __init__(self, data_path, tokenizer, model, max_length=512):
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length

        # Load data
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Get question and answer
        question = sample.get('question', sample.get('conversations', [{}])[0].get('value', ''))
        answer = sample.get('answer', sample.get('conversations', [{}])[1].get('value', ''))

        # Tokenize
        # For training, we use question as input to generate memory,
        # then evaluate if the enhanced representation helps answer generation
        question_ids = self.tokenizer(
            question,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        answer_ids = self.tokenizer(
            answer,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'question_input_ids': question_ids['input_ids'].squeeze(0),
            'question_attention_mask': question_ids['attention_mask'].squeeze(0),
            'answer_input_ids': answer_ids['input_ids'].squeeze(0),
            'answer_attention_mask': answer_ids['attention_mask'].squeeze(0),
            'image_path': sample.get('image', '')
        }


def train_memgen_weaver(args):
    """Train MemGen Weaver."""

    print("="*70)
    print("Training MemGen-style Memory Weaver")
    print("="*70)

    # === Setup ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load base model and tokenizer
    print(f"\nLoading base model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create MemGen Weaver
    print(f"\nCreating MemGen Weaver...")
    config = MemGenWeaverConfig(
        hidden_size=base_model.config.hidden_size,
        num_memory_tokens=args.num_memory_tokens,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )

    weaver = MemGenWeaver(base_model, config)
    weaver.to(device)

    print("\nTrainable parameters:")
    weaver.print_trainable_parameters()

    # Load dataset
    print(f"\nLoading training data from {args.train_data}...")
    train_dataset = MemoryWeaverDataset(
        args.train_data,
        tokenizer,
        base_model,
        max_length=args.max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in weaver.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Training loop
    print(f"\n{'='*70}")
    print("Starting Training")
    print(f"{'='*70}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"{'='*70}\n")

    weaver.train()
    global_step = 0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            question_ids = batch['question_input_ids'].to(device)
            question_mask = batch['question_attention_mask'].to(device)
            answer_ids = batch['answer_input_ids'].to(device)

            # === Forward pass ===
            # 1. Get input embeddings from question
            with torch.no_grad():
                question_embeds = base_model.get_input_embeddings()(question_ids)

            # 2. Generate memory tokens using weaver
            memory_tokens = weaver(
                inputs_embeds=question_embeds,
                attention_mask=question_mask
            )

            # 3. Prepare enhanced input for answer generation
            # Concatenate memory tokens with answer input
            answer_embeds = base_model.get_input_embeddings()(answer_ids)

            # For training, we prepend memory tokens to answer sequence
            batch_size = memory_tokens.size(0)
            enhanced_embeds = torch.cat([memory_tokens, answer_embeds], dim=1)

            # Update attention mask
            memory_mask = torch.ones(
                batch_size, memory_tokens.size(1),
                dtype=question_mask.dtype,
                device=device
            )
            enhanced_mask = torch.cat([memory_mask, batch['answer_attention_mask'].to(device)], dim=1)

            # 4. Forward through base model to compute loss
            outputs = base_model(
                inputs_embeds=enhanced_embeds,
                attention_mask=enhanced_mask,
                labels=answer_ids  # Only compute loss on answer part
            )

            loss = outputs.loss

            # === Backward pass ===
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in weaver.parameters() if p.requires_grad],
                max_norm=1.0
            )

            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
            })

            # Log
            if global_step % args.log_interval == 0:
                print(f"\nStep {global_step}: loss = {loss.item():.4f}")

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            weaver.save_adapter(save_path)
            print(f"  ✓ Saved checkpoint to {save_path}")

    # Final save
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    weaver.save_adapter(final_path)

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Train MemGen-style Memory Weaver")

    # Model arguments
    parser.add_argument('--model_path', type=str, default='/mnt/data/AirSpatialBot',
                        help='Path to base LLaVA model')

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data (JSONL format)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')

    # Memory Weaver arguments
    parser.add_argument('--num_memory_tokens', type=int, default=8,
                        help='Number of learnable query latents')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank (MemGen uses 16)')
    parser.add_argument('--lora_alpha', type=float, default=32.0,
                        help='LoRA alpha (MemGen uses 32)')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')

    # Logging and saving
    parser.add_argument('--output_dir', type=str, default='./outputs/memgen_weaver',
                        help='Output directory for checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (steps)')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save interval (epochs)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Train
    train_memgen_weaver(args)


if __name__ == '__main__':
    main()
