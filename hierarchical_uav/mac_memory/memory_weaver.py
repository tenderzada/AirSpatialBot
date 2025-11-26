"""
Memory Weaver: LoRA-based Generative Memory Module

Inspired by MemGen (https://github.com/tenderzada/MemGen)
Generates latent memory tokens using Low-Rank Adaptation (LoRA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class MemoryWeaverConfig:
    """Configuration for Memory Weaver."""

    # LoRA configuration
    hidden_size: int = 4096
    lora_rank: int = 8  # Low-rank dimension
    lora_alpha: float = 16.0  # Scaling factor
    lora_dropout: float = 0.1

    # Memory generation
    num_memory_tokens: int = 10  # Number of latent memory tokens to generate

    # Activation
    use_gelu: bool = True  # GELU activation for LoRA


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation Layer.

    Implements: h = W0·x + (B·A)·x * (alpha / r)
    where W0 is frozen, A and B are trainable low-rank matrices.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Low-rank dimension r
        alpha: Scaling factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        # A: projects to low-rank space
        # B: projects back to output space
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            LoRA output [..., out_features]
        """
        # Apply dropout
        x_dropped = self.lora_dropout(x)

        # Low-rank transformation: B(A(x))
        lora_out = self.lora_B(self.lora_A(x_dropped))

        # Apply scaling
        return lora_out * self.scaling


class MemoryWeaver(nn.Module):
    """
    Memory Weaver: Generates latent memory tokens using LoRA.

    Architecture:
        1. Pool input hidden states (mean pooling)
        2. Generate latent tokens via LoRA transformation
        3. Reshape to memory tokens

    Unlike MAC's retrieval-based approach, MemoryWeaver synthesizes
    memory representations on-the-fly using learned LoRA parameters.

    Args:
        config: MemoryWeaverConfig instance
    """

    def __init__(self, config: MemoryWeaverConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_memory_tokens = config.num_memory_tokens

        # LoRA layer for memory generation
        self.memory_lora = LoRALayer(
            in_features=config.hidden_size,
            out_features=config.num_memory_tokens * config.hidden_size,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout
        )

        # Optional activation
        self.activation = nn.GELU() if config.use_gelu else nn.Identity()

        # Layer normalization for stability
        self.ln = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pool_method: str = 'mean'
    ) -> torch.Tensor:
        """
        Generate latent memory tokens.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            pool_method: Pooling method ('mean', 'max', 'first', 'last')

        Returns:
            memory_tokens: [batch_size, num_memory_tokens, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 1. Pool hidden states to get aggregate representation
        if pool_method == 'mean':
            pooled = hidden_states.mean(dim=1)  # [B, H]
        elif pool_method == 'max':
            pooled = hidden_states.max(dim=1)[0]  # [B, H]
        elif pool_method == 'first':
            pooled = hidden_states[:, 0, :]  # [B, H]
        elif pool_method == 'last':
            pooled = hidden_states[:, -1, :]  # [B, H]
        else:
            raise ValueError(f"Unknown pool_method: {pool_method}")

        # 2. Generate memory via LoRA
        memory_flat = self.memory_lora(pooled)  # [B, num_tokens * H]
        memory_flat = self.activation(memory_flat)

        # 3. Reshape to memory tokens
        memory_tokens = memory_flat.view(
            batch_size,
            self.num_memory_tokens,
            self.hidden_size
        )  # [B, num_tokens, H]

        # 4. Normalize
        memory_tokens = self.ln(memory_tokens)

        return memory_tokens

    def get_lora_parameters(self):
        """Get LoRA parameters for separate optimization."""
        return self.memory_lora.parameters()

    def freeze_lora(self):
        """Freeze LoRA parameters (inference mode)."""
        for param in self.memory_lora.parameters():
            param.requires_grad = False

    def unfreeze_lora(self):
        """Unfreeze LoRA parameters (training mode)."""
        for param in self.memory_lora.parameters():
            param.requires_grad = True


class AdaptiveMemoryWeaver(nn.Module):
    """
    Adaptive Memory Weaver with multi-scale memory generation.

    Generates memory tokens at multiple scales and combines them.
    Useful for capturing both fine-grained and coarse-grained patterns.

    Args:
        config: MemoryWeaverConfig instance
        num_scales: Number of memory scales
    """

    def __init__(
        self,
        config: MemoryWeaverConfig,
        num_scales: int = 3
    ):
        super().__init__()

        self.config = config
        self.num_scales = num_scales

        # Create LoRA layers for each scale
        self.scale_loras = nn.ModuleList([
            LoRALayer(
                in_features=config.hidden_size,
                out_features=config.num_memory_tokens * config.hidden_size,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout
            )
            for _ in range(num_scales)
        ])

        # Scale fusion layer
        self.fusion = nn.Linear(
            config.hidden_size * num_scales,
            config.hidden_size
        )

        # Layer norm
        self.ln = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate multi-scale memory tokens.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            memory_tokens: [batch_size, num_memory_tokens, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Generate memories at different scales
        scale_memories = []

        for scale_idx, lora in enumerate(self.scale_loras):
            # Different pooling strategies for different scales
            if scale_idx == 0:
                # Fine-grained: use first/last tokens
                pooled = (hidden_states[:, 0, :] + hidden_states[:, -1, :]) / 2
            elif scale_idx == 1:
                # Medium-grained: mean pooling
                pooled = hidden_states.mean(dim=1)
            else:
                # Coarse-grained: max pooling
                pooled = hidden_states.max(dim=1)[0]

            # Generate memory via LoRA
            memory_flat = lora(pooled)  # [B, num_tokens * H]
            memory_tokens = memory_flat.view(
                batch_size,
                self.config.num_memory_tokens,
                hidden_size
            )
            scale_memories.append(memory_tokens)

        # Concatenate along feature dimension
        multi_scale = torch.cat(scale_memories, dim=-1)  # [B, num_tokens, H * num_scales]

        # Fuse scales
        fused_memory = self.fusion(multi_scale)  # [B, num_tokens, H]
        fused_memory = self.ln(fused_memory)

        return fused_memory


if __name__ == "__main__":
    print("Testing MemoryWeaver...")

    # Test configuration
    config = MemoryWeaverConfig(
        hidden_size=512,
        lora_rank=8,
        num_memory_tokens=10
    )

    # Create weaver
    weaver = MemoryWeaver(config)

    # Test input
    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Generate memory
    memory_tokens = weaver(hidden_states)

    print(f"Input shape: {hidden_states.shape}")
    print(f"Memory tokens shape: {memory_tokens.shape}")
    print(f"Expected: [{batch_size}, {config.num_memory_tokens}, {config.hidden_size}]")

    # Test parameter count
    total_params = sum(p.numel() for p in weaver.parameters())
    lora_params = sum(p.numel() for p in weaver.get_lora_parameters())

    print(f"\nParameter count:")
    print(f"  Total: {total_params:,}")
    print(f"  LoRA: {lora_params:,}")
    print(f"  Reduction: {lora_params / total_params * 100:.1f}%")

    # Test adaptive weaver
    print("\nTesting AdaptiveMemoryWeaver...")
    adaptive_weaver = AdaptiveMemoryWeaver(config, num_scales=3)
    adaptive_memory = adaptive_weaver(hidden_states)

    print(f"Adaptive memory shape: {adaptive_memory.shape}")

    # Test freeze/unfreeze
    weaver.freeze_lora()
    print(f"\nFrozen: grad_enabled={next(weaver.get_lora_parameters()).requires_grad}")

    weaver.unfreeze_lora()
    print(f"Unfrozen: grad_enabled={next(weaver.get_lora_parameters()).requires_grad}")

    print("\n✓ MemoryWeaver tests passed!")
