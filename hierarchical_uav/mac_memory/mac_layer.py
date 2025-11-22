"""
MAC Layer: Memory-Augmented Continual Learning Layer

Integrates neural memory, persistent memory, and attention mechanism
for test-time learning capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from .neural_memory import NeuralMemory
from .persistent_memory import PersistentMemory


@dataclass
class MACConfig:
    """Configuration for MAC Layer."""

    # Attention configuration
    hidden_size: int = 4096
    num_attention_heads: int = 32
    attention_dropout: float = 0.1

    # Memory configuration
    memory_dim: int = 4096
    memory_depth: int = 2
    num_persistent_tokens: int = 64
    num_memory_tokens: int = 128

    # Segment processing
    segment_size: int = 512  # Process input in segments

    # Learning configuration
    enable_memory_update: bool = True  # Set False for L-UAV
    surprise_eta: float = 0.9   # Surprise decay
    learning_theta: float = 0.1  # Learning rate
    forgetting_alpha: float = 0.01  # Weight decay


class MACLayer(nn.Module):
    """
    MAC Layer combining memory and attention.

    Architecture:
        1. Split input into segments
        2. For each segment:
           a. Retrieve from long-term memory
           b. Concatenate: persistent_memory || retrieved_memory || segment
           c. Apply attention
           d. Update long-term memory (if enabled)

    Args:
        config: MACConfig instance
    """

    def __init__(self, config: MACConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        assert (
            self.head_dim * config.num_attention_heads == config.hidden_size
        ), "hidden_size must be divisible by num_attention_heads"

        # Long-term memory (neural memory)
        self.neural_memory = NeuralMemory(
            input_dim=config.memory_dim,
            output_dim=config.memory_dim,
            hidden_dim=config.memory_dim,
            num_layers=config.memory_depth
        )

        # Persistent memory (learnable context)
        self.persistent_memory = PersistentMemory(
            num_tokens=config.num_persistent_tokens,
            token_dim=config.hidden_size
        )

        # Q, K, V projections for attention
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Memory query projection
        self.memory_query_proj = nn.Linear(config.hidden_size, config.memory_dim)

        # Memory value projection (for retrieved tokens)
        self.memory_value_proj = nn.Linear(
            config.memory_dim,
            config.hidden_size * config.num_memory_tokens
        )

        # Layer norm
        self.ln_pre = nn.LayerNorm(config.hidden_size)
        self.ln_post = nn.LayerNorm(config.hidden_size)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def _split_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Split last dimension into (num_heads, head_dim).

        Args:
            x: [batch_size, seq_len, hidden_size]

        Returns:
            [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge (num_heads, head_dim) into last dimension.

        Args:
            x: [batch_size, num_heads, seq_len, head_dim]

        Returns:
            [batch_size, seq_len, hidden_size]
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.hidden_size)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Scaled dot-product attention.

        Args:
            q: [batch_size, num_heads, q_len, head_dim]
            k: [batch_size, num_heads, k_len, head_dim]
            v: [batch_size, num_heads, v_len, head_dim]
            attention_mask: Optional mask

        Returns:
            output: [batch_size, num_heads, q_len, head_dim]
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        return output

    def retrieve_from_memory(
        self,
        query_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieve tokens from long-term memory.

        Args:
            query_features: [batch_size, hidden_size] aggregate query

        Returns:
            memory_tokens: [batch_size, num_memory_tokens, hidden_size]
        """
        batch_size = query_features.shape[0]

        # Project to memory query space
        memory_query = self.memory_query_proj(query_features)  # [B, memory_dim]

        # Retrieve from neural memory
        memory_values = []
        for i in range(batch_size):
            value, _ = self.neural_memory.retrieve_with_cache(memory_query[i])
            memory_values.append(value)

        memory_values = torch.stack(memory_values, dim=0)  # [B, memory_dim]

        # Project to token space
        memory_tokens_flat = self.memory_value_proj(memory_values)  # [B, hidden*num_tokens]

        # Reshape to tokens
        memory_tokens = memory_tokens_flat.view(
            batch_size,
            self.config.num_memory_tokens,
            self.hidden_size
        )

        return memory_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with memory-augmented attention.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            update_memory: Override config setting for memory update

        Returns:
            output: [batch_size, seq_len, hidden_size]
            metrics: Dictionary with memory statistics
        """
        if update_memory is None:
            update_memory = self.config.enable_memory_update

        # Ensure float32 for numerical stability in test-time learning
        # This is critical when model uses mixed precision (8-bit loading)
        hidden_states = hidden_states.float()

        batch_size, seq_len, hidden_size = hidden_states.shape

        metrics = {
            'memory_loss': 0.0,
            'surprise': 0.0,
            'cache_hits': 0
        }

        # Pre-layer norm
        x = self.ln_pre(hidden_states)

        # 1. Retrieve from long-term memory
        # Use mean-pooled features as query
        query_features = x.mean(dim=1)  # [batch_size, hidden_size]
        memory_tokens = self.retrieve_from_memory(query_features)

        # 2. Get persistent memory
        persistent_tokens = self.persistent_memory(batch_size)

        # 3. Concatenate: persistent || memory || input
        context = torch.cat([
            persistent_tokens,  # [B, Np, H]
            memory_tokens,      # [B, Nm, H]
            x                   # [B, seq_len, H]
        ], dim=1)

        # 4. Apply attention
        q = self._split_heads(self.q_proj(x))  # [B, num_heads, seq_len, head_dim]
        k = self._split_heads(self.k_proj(context))
        v = self._split_heads(self.v_proj(context))

        attn_output = self._attention(q, k, v, attention_mask)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        # Residual connection
        output = hidden_states + attn_output
        output = self.ln_post(output)

        # 5. Update long-term memory (if enabled)
        # IMPORTANT: Test-time learning works in eval mode!
        # We update memory during inference to continuously adapt
        if update_memory:
            # Use output features as target for memory update
            target_features = output.mean(dim=1)  # [B, H]
            target_memory = self.memory_query_proj(target_features)  # [B, memory_dim]

            for i in range(batch_size):
                update_metrics = self.neural_memory.update_memory(
                    query=self.memory_query_proj(query_features[i]),
                    target_value=target_memory[i],
                    eta=self.config.surprise_eta,
                    theta=self.config.learning_theta,
                    alpha=self.config.forgetting_alpha
                )

                metrics['memory_loss'] += update_metrics['loss']
                metrics['surprise'] += update_metrics['surprise']

            # Average metrics
            metrics['memory_loss'] /= batch_size
            metrics['surprise'] /= batch_size

        return output, metrics

    def freeze_for_luav(self):
        """
        Freeze all parameters for L-UAV (inference-only mode).
        """
        for param in self.parameters():
            param.requires_grad = False

        self.config.enable_memory_update = False

    def unfreeze_for_huav(self):
        """
        Unfreeze parameters for H-UAV (learning mode).
        """
        for param in self.parameters():
            param.requires_grad = True

        self.config.enable_memory_update = True


if __name__ == "__main__":
    print("Testing MACLayer...")

    config = MACConfig(
        hidden_size=512,
        num_attention_heads=8,
        memory_dim=512,
        num_persistent_tokens=16,
        num_memory_tokens=32,
        segment_size=128
    )

    mac_layer = MACLayer(config)

    # Test forward pass
    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    output, metrics = mac_layer(hidden_states, update_memory=False)

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Metrics: {metrics}")

    # Test freeze/unfreeze
    mac_layer.freeze_for_luav()
    print(f"Frozen (L-UAV): grad_enabled={next(mac_layer.parameters()).requires_grad}")

    mac_layer.unfreeze_for_huav()
    print(f"Unfrozen (H-UAV): grad_enabled={next(mac_layer.parameters()).requires_grad}")

    print("✓ MACLayer tests passed!")
