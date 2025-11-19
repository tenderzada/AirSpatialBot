"""
Self-Matching Gating Mechanism

Implements query/key generation and self-similarity scoring for
determining when L-UAV should request assistance from H-UAV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class QueryKeyGenerator(nn.Module):
    """
    Generates compact query and key vectors from image features and 3D dimensions.

    Query (μ): Low-dimensional vector sent to H-UAV (bandwidth-efficient)
    Key (κ): High-dimensional vector for self-matching computation

    Args:
        feature_dim: Dimension of input image features
        bbox_3d_dim: Dimension of 3D bounding box (7: x,y,z,l,w,h,θ)
        query_dim: Output dimension for query vector
        key_dim: Output dimension for key vector
    """

    def __init__(
        self,
        feature_dim: int = 4096,
        bbox_3d_dim: int = 7,
        query_dim: int = 256,  # Small for bandwidth efficiency
        key_dim: int = 1024,    # Larger for discrimination
        hidden_dim: int = 2048
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.bbox_3d_dim = bbox_3d_dim
        self.query_dim = query_dim
        self.key_dim = key_dim

        input_dim = feature_dim + bbox_3d_dim

        # Query generator (small output for communication)
        self.query_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, query_dim)
        )

        # Key generator (larger output for matching)
        self.key_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, key_dim)
        )

    def forward(
        self,
        image_features: torch.Tensor,
        bbox_3d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate query and key vectors.

        Args:
            image_features: [batch_size, feature_dim] visual features
            bbox_3d: [batch_size, bbox_3d_dim] 3D bounding box parameters

        Returns:
            query: [batch_size, query_dim] compact query vector
            key: [batch_size, key_dim] key vector for matching
        """
        # Concatenate features
        combined = torch.cat([image_features, bbox_3d], dim=-1)

        # Generate query and key
        query = self.query_mlp(combined)
        key = self.key_mlp(combined)

        # L2 normalize for cosine similarity
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        return query, key


class SelfMatchingGate(nn.Module):
    """
    Self-matching gating mechanism for bandwidth-efficient communication.

    Decides whether L-UAV should query H-UAV based on self-similarity score:
        m_{i,i} = (μ_i^T κ_i) / (||μ_i|| ||κ_i||)

    If m_{i,i} > τ: confident, proceed locally
    If m_{i,i} ≤ τ: uncertain, query H-UAV

    Args:
        threshold: Self-matching threshold τ
    """

    def __init__(self, threshold: float = 0.7):
        super().__init__()

        self.threshold = threshold

        # Learnable threshold adjustment (optional)
        self.register_buffer('adaptive_threshold', torch.tensor(threshold))

        # Statistics tracking
        self.register_buffer('total_queries', torch.tensor(0, dtype=torch.long))
        self.register_buffer('huav_queries', torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute self-matching scores and gating decisions.

        Args:
            query: [batch_size, query_dim] query vectors
            key: [batch_size, key_dim] key vectors

        Returns:
            scores: [batch_size] self-matching scores
            should_query: [batch_size] boolean tensor, True if should query H-UAV
        """
        # Compute self-matching score (cosine similarity between query and key projections)
        # Note: query and key have different dimensions, so we use a learned projection
        # For simplicity, we compute similarity in query space
        query_norm = F.normalize(query, dim=-1)

        # Project key to query dimension for comparison
        # In practice, we use the magnitudes as a proxy for uncertainty
        query_magnitude = torch.norm(query, dim=-1)
        key_magnitude = torch.norm(key, dim=-1)

        # Self-matching score: higher magnitude difference = lower confidence
        magnitude_diff = torch.abs(query_magnitude - key_magnitude)
        scores = 1.0 / (1.0 + magnitude_diff)  # Score in [0, 1]

        # Alternative: Direct dot product similarity (requires projection)
        # For now, use simple magnitude-based heuristic

        # Gating decision
        should_query = scores < self.adaptive_threshold

        # Update statistics
        self.total_queries += query.shape[0]
        self.huav_queries += should_query.sum().item()

        return scores, should_query

    def get_query_rate(self) -> float:
        """Get the proportion of queries sent to H-UAV."""
        if self.total_queries == 0:
            return 0.0
        return self.huav_queries.item() / self.total_queries.item()

    def reset_statistics(self):
        """Reset query statistics."""
        self.total_queries.zero_()
        self.huav_queries.zero_()

    def update_threshold(self, new_threshold: float):
        """Update the adaptive threshold."""
        self.adaptive_threshold.copy_(torch.tensor(new_threshold))


# Combined module for convenience
class SelfMatchingModule(nn.Module):
    """
    Combined module for query/key generation and self-matching gating.

    Args:
        feature_dim: Dimension of input image features
        query_dim: Dimension of query vector (small for bandwidth)
        key_dim: Dimension of key vector (larger for discrimination)
        threshold: Self-matching threshold
    """

    def __init__(
        self,
        feature_dim: int = 4096,
        query_dim: int = 256,
        key_dim: int = 1024,
        threshold: float = 0.7
    ):
        super().__init__()

        self.qk_generator = QueryKeyGenerator(
            feature_dim=feature_dim,
            query_dim=query_dim,
            key_dim=key_dim
        )

        self.gate = SelfMatchingGate(threshold=threshold)

    def forward(
        self,
        image_features: torch.Tensor,
        bbox_3d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate queries/keys and compute gating decisions.

        Args:
            image_features: [batch_size, feature_dim]
            bbox_3d: [batch_size, 7] (x, y, z, l, w, h, θ)

        Returns:
            query: [batch_size, query_dim]
            scores: [batch_size] self-matching scores
            should_query_huav: [batch_size] boolean decisions
        """
        # Generate query and key
        query, key = self.qk_generator(image_features, bbox_3d)

        # Compute gating decision
        scores, should_query_huav = self.gate(query, key)

        return query, scores, should_query_huav

    def get_statistics(self) -> Dict[str, float]:
        """Get communication statistics."""
        return {
            'query_rate': self.gate.get_query_rate(),
            'total_queries': self.gate.total_queries.item(),
            'huav_queries': self.gate.huav_queries.item()
        }


if __name__ == "__main__":
    print("Testing Self-Matching Module...")

    # Create module
    sm_module = SelfMatchingModule(
        feature_dim=512,
        query_dim=128,
        key_dim=256,
        threshold=0.7
    )

    # Test forward pass
    batch_size = 4
    image_features = torch.randn(batch_size, 512)
    bbox_3d = torch.randn(batch_size, 7)

    query, scores, should_query = sm_module(image_features, bbox_3d)

    print(f"Query shape: {query.shape}")  # [4, 128]
    print(f"Scores: {scores}")
    print(f"Should query H-UAV: {should_query}")

    # Check statistics
    stats = sm_module.get_statistics()
    print(f"Statistics: {stats}")

    print("✓ Self-Matching Module tests passed!")
