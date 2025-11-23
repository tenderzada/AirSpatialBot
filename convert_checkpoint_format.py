#!/usr/bin/env python3
"""
Convert old checkpoint format to new format.

Old format:
- neural_memory_state
- persistent_memory_state

New format:
- neural_memory
- persistent_memory
- cache
- surprise
- step_count
"""

import torch
import sys
import os
from pathlib import Path

def convert_checkpoint(old_path, new_path=None):
    """Convert checkpoint from old format to new format."""

    if new_path is None:
        # Create backup and overwrite original
        backup_path = old_path + '.backup'
        new_path = old_path
        print(f"Creating backup: {backup_path}")
        import shutil
        shutil.copy2(old_path, backup_path)

    print(f"Loading old checkpoint: {old_path}")
    checkpoint = torch.load(old_path, map_location='cpu')

    print("Old checkpoint keys:", list(checkpoint.keys()))

    # Check if already in new format
    if 'neural_memory' in checkpoint:
        print("✓ Checkpoint is already in new format!")
        return

    # Convert to new format
    new_checkpoint = {
        'epoch': checkpoint.get('epoch', 0),
        'stats': checkpoint.get('stats', [])
    }

    # Copy MAC state as-is
    if 'mac_state' in checkpoint:
        new_checkpoint['mac_state'] = checkpoint['mac_state']

    # Rename neural_memory_state -> neural_memory
    if 'neural_memory_state' in checkpoint:
        new_checkpoint['neural_memory'] = checkpoint['neural_memory_state']
        print("✓ Converted 'neural_memory_state' -> 'neural_memory'")

    # Rename persistent_memory_state -> persistent_memory
    if 'persistent_memory_state' in checkpoint:
        new_checkpoint['persistent_memory'] = checkpoint['persistent_memory_state']
        print("✓ Converted 'persistent_memory_state' -> 'persistent_memory'")

    # Add missing cache, surprise, step_count with default values
    # These will be initialized when loading the model
    new_checkpoint['cache'] = {
        'keys': [],
        'values': [],
        'timestamps': []
    }
    new_checkpoint['surprise'] = torch.zeros(1)
    new_checkpoint['step_count'] = torch.tensor(0, dtype=torch.long)

    print("✓ Added 'cache', 'surprise', 'step_count' with default values")

    # Save converted checkpoint
    print(f"Saving new checkpoint: {new_path}")
    torch.save(new_checkpoint, new_path)

    print("\n" + "="*60)
    print("✓ Conversion complete!")
    print("="*60)
    print(f"New checkpoint keys: {list(new_checkpoint.keys())}")

    # Show file sizes
    old_size = os.path.getsize(old_path) / (1024**3)
    new_size = os.path.getsize(new_path) / (1024**3)
    print(f"\nOld size: {old_size:.2f} GB")
    print(f"New size: {new_size:.2f} GB")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_checkpoint_format.py <checkpoint_path> [output_path]")
        print("\nExample:")
        print("  python convert_checkpoint_format.py ./outputs/huav_training/huav_memory_final.pt")
        print("  (This will backup and overwrite the original file)")
        print("\nOr:")
        print("  python convert_checkpoint_format.py old.pt new.pt")
        print("  (This will create a new file)")
        sys.exit(1)

    old_path = sys.argv[1]
    new_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(old_path):
        print(f"Error: File not found: {old_path}")
        sys.exit(1)

    convert_checkpoint(old_path, new_path)
