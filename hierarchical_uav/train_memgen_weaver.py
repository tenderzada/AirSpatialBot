"""
Training script for MemGen-style Memory Weaver

Features:
- Validation split
- Best checkpoint saving (based on validation loss)
- Loss curve plotting
- TensorBoard logging
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

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

        # Get question and answer - handle multiple data formats
        question = ''
        answer = ''

        # Format 1: Direct question/answer fields
        if 'question' in sample:
            question = sample['question']
            answer = sample.get('answer', '')
        # Format 2: Conversations format
        elif 'conversations' in sample and len(sample['conversations']) >= 2:
            question = sample['conversations'][0].get('value', '')
            answer = sample['conversations'][1].get('value', '')
        # Format 3: Single conversation (use as question, empty answer)
        elif 'conversations' in sample and len(sample['conversations']) >= 1:
            question = sample['conversations'][0].get('value', '')
            answer = ''
        else:
            # Fallback: use any text field
            question = str(sample.get('text', sample.get('prompt', '')))
            answer = str(sample.get('response', sample.get('completion', '')))

        # Tokenize
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


def validate(weaver, base_model, val_loader, device):
    """Run validation and return average loss."""
    weaver.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            # Move to device
            question_ids = batch['question_input_ids'].to(device)
            question_mask = batch['question_attention_mask'].to(device)
            answer_ids = batch['answer_input_ids'].to(device)

            # Get input embeddings
            question_embeds = base_model.get_input_embeddings()(question_ids)

            # Generate memory tokens
            memory_tokens = weaver(
                inputs_embeds=question_embeds,
                attention_mask=question_mask
            )

            # Enhanced input for answer generation
            answer_embeds = base_model.get_input_embeddings()(answer_ids)
            batch_size = memory_tokens.size(0)
            enhanced_embeds = torch.cat([memory_tokens, answer_embeds], dim=1)

            # Update attention mask
            memory_mask = torch.ones(
                batch_size, memory_tokens.size(1),
                dtype=question_mask.dtype,
                device=device
            )
            enhanced_mask = torch.cat([memory_mask, batch['answer_attention_mask'].to(device)], dim=1)

            # Forward through base model
            outputs = base_model(
                inputs_embeds=enhanced_embeds,
                attention_mask=enhanced_mask,
                labels=answer_ids
            )

            total_loss += outputs.loss.item()
            num_batches += 1

    weaver.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


def plot_losses(train_losses, val_losses, save_path):
    """Plot and save loss curves."""
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('MemGen Weaver Training Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add best validation loss annotation
    best_val_epoch = val_losses.index(min(val_losses)) + 1
    best_val_loss = min(val_losses)
    plt.axvline(x=best_val_epoch, color='g', linestyle='--', alpha=0.5)
    plt.text(best_val_epoch, best_val_loss,
             f'Best: {best_val_loss:.4f}\nEpoch {best_val_epoch}',
             ha='left', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Loss curve saved to {save_path}")


def train_memgen_weaver(args):
    """Train MemGen Weaver."""

    print("="*70)
    print("Training MemGen-style Memory Weaver")
    print("="*70)

    # === Setup ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    # Load base model and tokenizer
    print(f"\nLoading base model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load LLaVA model (handle both LLaVA and standard LLM)
    try:
        # Try loading as LLaVA first
        from transformers import LlavaForConditionalGeneration
        base_model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        print("Loaded as LLaVA model")
        # For LLaVA, we work with the language model part
        if hasattr(base_model, 'language_model'):
            base_model = base_model.language_model
            print("Using language_model component")
    except Exception as e:
        print(f"LLaVA loading failed ({e}), trying AutoModel...")
        # Fallback to AutoModel
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True
        )
        print("Loaded with AutoModel")

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
        target_modules=["q_proj", "v_proj"],
        target_layers=[8, 16, 24]  # Inject LoRA in these layers
    )

    weaver = MemGenWeaver(base_model, config)
    weaver.to(device)

    print("\nTrainable parameters:")
    weaver.print_trainable_parameters()

    # Load full dataset
    print(f"\nLoading data from {args.train_data}...")
    full_dataset = MemoryWeaverDataset(
        args.train_data,
        tokenizer,
        base_model,
        max_length=args.max_length
    )

    # Split into train and validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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
    print(f"Validation Split: {args.val_split}")
    print(f"{'='*70}\n")

    weaver.train()
    global_step = 0
    best_val_loss = float('inf')
    best_epoch = 0

    # Loss tracking
    train_losses = []
    val_losses = []

    for epoch in range(args.num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*70}")

        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training")

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
            answer_embeds = base_model.get_input_embeddings()(answer_ids)
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
                labels=answer_ids
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

            # TensorBoard logging
            writer.add_scalar('Train/Loss', loss.item(), global_step)

        # Epoch training summary
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f"\n📊 Epoch {epoch + 1} Training Summary:")
        print(f"  Average Training Loss: {avg_train_loss:.4f}")

        # Validation
        print(f"\n🔍 Running validation...")
        val_loss = validate(weaver, base_model, val_loader, device)
        val_losses.append(val_loss)

        print(f"  Validation Loss: {val_loss:.4f}")

        # TensorBoard logging
        writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            best_path = os.path.join(args.output_dir, "best_checkpoint")
            os.makedirs(best_path, exist_ok=True)
            weaver.save_adapter(best_path)

            # Save metadata
            with open(os.path.join(best_path, 'metadata.json'), 'w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'global_step': global_step
                }, f, indent=2)

            print(f"\n  ✅ New best checkpoint saved! (val_loss: {val_loss:.4f})")

        print(f"\n  📌 Best so far: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")

    # Training complete
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"Best checkpoint: {os.path.join(args.output_dir, 'best_checkpoint')}")

    # Plot loss curves
    loss_plot_path = os.path.join(args.output_dir, 'loss_curves.png')
    plot_losses(train_losses, val_losses, loss_plot_path)

    # Save loss history
    loss_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }

    with open(os.path.join(args.output_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f, indent=2)

    print(f"\n✓ Loss history saved to {args.output_dir}/loss_history.json")

    # Close TensorBoard writer
    writer.close()

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Train MemGen-style Memory Weaver")

    # Model arguments
    parser.add_argument('--model_path', type=str, default='/mnt/data/AirSpatialBot',
                        help='Path to base LLaVA model')

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data (JSONL format)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
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
