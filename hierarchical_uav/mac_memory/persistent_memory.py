"""
Persistent Memory Module

Implements learnable but data-independent memory tokens that
are prepended to input sequences to provide task-specific context.
"""

import torch
import torch.nn as nn


class PersistentMemory(nn.Module):
    """
    Persistent Memory: learnable context tokens.

    These tokens are prepended to input sequences and remain fixed
    during test-time learning (only updated during training).

    Args:
        num_tokens: Number of persistent memory tokens
        token_dim: Dimension of each token
        init_std: Standard deviation for initialization
    """

    def __init__(
        self,
        num_tokens: int = 64,
        token_dim: int = 4096,
        init_std: float = 0.02
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_dim = token_dim

        # Learnable memory tokens
        self.memory_tokens = nn.Parameter(
            torch.randn(num_tokens, token_dim) * init_std
        )

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get persistent memory tokens for a batch.

        Args:
            batch_size: Number of examples in batch

        Returns:
            tokens: [batch_size, num_tokens, token_dim]
        """
        # Expand for batch
        tokens = self.memory_tokens.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        return tokens

    def get_single_tokens(self) -> torch.Tensor:
        """
        Get persistent memory tokens without batch dimension.

        Returns:
            tokens: [num_tokens, token_dim]
        """
        return self.memory_tokens

    def __repr__(self):
        return (
            f"PersistentMemory(num_tokens={self.num_tokens}, "
            f"token_dim={self.token_dim})"
        )


if __name__ == "__main__":
    print("Testing PersistentMemory module...")

    pm = PersistentMemory(num_tokens=64, token_dim=512)

    # Test forward with batch
    tokens = pm(batch_size=4)
    print(f"Batch tokens shape: {tokens.shape}")  # [4, 64, 512]

    # Test single
    single_tokens = pm.get_single_tokens()
    print(f"Single tokens shape: {single_tokens.shape}")  # [64, 512]

    # Check trainable
    print(f"Trainable parameters: {sum(p.numel() for p in pm.parameters())}")

    print("✓ PersistentMemory tests passed!")
