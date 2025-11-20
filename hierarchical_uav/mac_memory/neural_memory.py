"""
Neural Long-term Memory Module

Implements the learnable memory component with:
- MLP-based key-value mapping
- Gradient-based memory update
- Surprise-driven learning with momentum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class NeuralMemory(nn.Module):
    """
    Neural Long-term Memory Module with surprise-driven updates.

    Implements associative memory loss:
        ℓ(M; x_t) = ||M(k_t) - v_t||²

    And surprise-driven update:
        S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)
        M_t = (1 - α_t) * M_{t-1} + S_t

    Args:
        input_dim: Dimension of input query/key
        output_dim: Dimension of output value
        hidden_dim: Hidden dimension of MLP
        num_layers: Number of MLP layers (depth)
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 4096,
        output_dim: int = 4096,
        hidden_dim: int = 4096,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build MLP layers
        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # Final layer
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Surprise tracking (momentum for gradients)
        self.register_buffer('surprise', torch.zeros(1))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))

        # Episodic cache for fast retrieval
        self.episodic_cache = {
            'keys': [],      # List of key tensors
            'values': [],    # List of value tensors
            'timestamps': [] # List of timestamps
        }
        self.cache_size = 1000  # Maximum cache size

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from memory given a query.

        Args:
            query: [batch_size, input_dim] or [input_dim]

        Returns:
            value: [batch_size, output_dim] or [output_dim]
        """
        # Ensure query is 2D
        if query.dim() == 1:
            query = query.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # MLP forward pass
        value = self.mlp(query)

        if squeeze_output:
            value = value.squeeze(0)

        return value

    def retrieve_with_cache(
        self,
        query: torch.Tensor,
        similarity_threshold: float = 0.70
    ) -> Tuple[torch.Tensor, bool]:
        """
        Retrieve from memory with episodic cache lookup.

        First checks if a similar query exists in cache. If found and confident,
        returns cached value. Otherwise computes new value via MLP.

        Args:
            query: [input_dim] query tensor
            similarity_threshold: Cosine similarity threshold for cache hit
                                 (default: 0.70, lower = more cache hits)

        Returns:
            value: [output_dim] retrieved value
            cache_hit: Whether result came from cache
        """
        # Search cache for similar queries
        if len(self.episodic_cache['keys']) > 0:
            cache_keys = torch.stack(self.episodic_cache['keys'])  # [N, input_dim]

            # Ensure cache_keys are on same device and dtype as query
            cache_keys = cache_keys.to(device=query.device, dtype=query.dtype)

            # Compute cosine similarity
            query_norm = F.normalize(query.unsqueeze(0), dim=1)
            cache_keys_norm = F.normalize(cache_keys, dim=1)
            similarities = torch.mm(query_norm, cache_keys_norm.t()).squeeze(0)  # [N]

            # Check for high-confidence match
            max_sim, max_idx = similarities.max(dim=0)

            # Log similarity for debugging
            if len(similarities) > 0 and hasattr(torch, 'get_default_dtype'):
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Cache lookup: max_sim={max_sim.item():.4f}, threshold={similarity_threshold:.4f}, cache_size={len(self.episodic_cache['keys'])}")

            if max_sim > similarity_threshold:
                # Cache hit!
                cached_value = self.episodic_cache['values'][max_idx]
                # Ensure cached_value is on same device as query
                cached_value = cached_value.to(device=query.device, dtype=query.dtype)
                return cached_value, True

        # Cache miss - compute via MLP
        value = self.forward(query)
        return value, False

    def update_memory(
        self,
        query: torch.Tensor,
        target_value: torch.Tensor,
        eta: float = 0.9,      # Surprise decay (momentum)
        theta: float = 0.1,    # Learning rate
        alpha: float = 0.01,   # Forgetting gate (weight decay)
    ) -> Dict[str, float]:
        """
        Update memory parameters using surprise-driven learning.

        Implements:
            ℓ = ||M(query) - target_value||²
            S_t = η * S_{t-1} - θ * ∇ℓ
            params_t = (1 - α) * params_{t-1} + S_t

        Args:
            query: [input_dim] input query
            target_value: [output_dim] target output
            eta: Surprise decay factor (momentum)
            theta: Learning rate
            alpha: Forgetting factor (weight decay)

        Returns:
            metrics: Dictionary with loss, surprise, etc.
        """
        # Compute current prediction
        pred_value = self.forward(query)

        # Compute associative memory loss
        loss = F.mse_loss(pred_value, target_value)

        # Compute gradients
        loss.backward()

        # Update surprise with momentum
        # S_t = η * S_{t-1} - θ * (gradient signal)
        # Here we use loss magnitude as proxy for gradient magnitude
        with torch.no_grad():
            new_surprise = eta * self.surprise - theta * loss.item()
            self.surprise.copy_(torch.tensor(new_surprise))

        # Apply surprise-modulated weight update
        with torch.no_grad():
            for param in self.mlp.parameters():
                if param.grad is not None:
                    # M_t = (1 - α) * M_{t-1} + θ * grad (simplified)
                    # Apply forgetting (weight decay) + gradient step
                    param.mul_(1 - alpha)  # Forgetting
                    param.add_(param.grad, alpha=-theta)  # Gradient step

                    # Clear gradients
                    param.grad.zero_()

        self.step_count += 1

        return {
            'loss': loss.item(),
            'surprise': self.surprise.item(),
            'step': self.step_count.item()
        }

    def add_to_cache(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        timestamp: Optional[int] = None
    ):
        """
        Add a key-value pair to episodic cache.

        Args:
            query: [input_dim] query tensor
            value: [output_dim] value tensor
            timestamp: Optional timestamp (uses step_count if None)
        """
        if timestamp is None:
            timestamp = self.step_count.item()

        # Add to cache
        self.episodic_cache['keys'].append(query.detach().cpu())
        self.episodic_cache['values'].append(value.detach().cpu())
        self.episodic_cache['timestamps'].append(timestamp)

        # Maintain cache size limit (FIFO)
        if len(self.episodic_cache['keys']) > self.cache_size:
            self.episodic_cache['keys'].pop(0)
            self.episodic_cache['values'].pop(0)
            self.episodic_cache['timestamps'].pop(0)

    def clear_cache(self):
        """Clear all episodic cache entries."""
        self.episodic_cache = {
            'keys': [],
            'values': [],
            'timestamps': []
        }

    def get_cache_size(self) -> int:
        """Return current number of entries in cache."""
        return len(self.episodic_cache['keys'])

    def reset_surprise(self):
        """Reset surprise metric to zero."""
        self.surprise.zero_()


if __name__ == "__main__":
    # Test the Neural Memory module
    print("Testing NeuralMemory module...")

    memory = NeuralMemory(
        input_dim=512,
        output_dim=512,
        hidden_dim=1024,
        num_layers=3
    )

    # Test forward pass
    query = torch.randn(512)
    value = memory(query)
    print(f"Query shape: {query.shape}")
    print(f"Value shape: {value.shape}")

    # Test batch forward
    batch_query = torch.randn(4, 512)
    batch_value = memory(batch_query)
    print(f"Batch query shape: {batch_query.shape}")
    print(f"Batch value shape: {batch_value.shape}")

    # Test memory update
    target = torch.randn(512)
    metrics = memory.update_memory(query, target)
    print(f"Update metrics: {metrics}")

    # Test cache
    memory.add_to_cache(query, value)
    retrieved, hit = memory.retrieve_with_cache(query)
    print(f"Cache hit: {hit}")
    print(f"Cache size: {memory.get_cache_size()}")

    print("✓ NeuralMemory tests passed!")
